"""
CAMELS-US Dataset for HydroArray

Large-sample hydrology dataset for the contiguous USA, containing
meteorological forcing data, streamflow observations, and catchment attributes
for 671 basins.

This module provides a simple interface that can be used directly as
a PyTorch Dataset for ML training.

Example:
    >>> from HydroArray.datasets import CAMELSUSDataset
    >>> from HydroArray.ml import train

    >>> # Simple usage - 3 lines to train
    >>> dataset = CAMELSUSDataset(
    ...     data_dir="path/to/camels",
    ...     basins=["01013500", "01015000"],
    ...     seq_len=365
    ... )
    >>> model, results = train(dataset, model="lstm", epochs=50)

    >>> # Or use as PyTorch Dataset
    >>> loader = dataset.to_dataloader(batch_size=32)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import cached_property

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from HydroArray.datasets.basin.base import BasinDataset, BasinData


CAMELS_US_FORCING_VARS = ['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']
CAMELS_US_TARGET_VAR = 'QObs(mm/d)'


@BasinDataset.register("camels_us")
class CAMELSUSDataset(BasinDataset, Dataset):
    """
    CAMELS-US dataset that can be used directly as a PyTorch Dataset.

    This class inherits from both BasinDataset and torch.utils.data.Dataset,
    allowing it to be used directly with PyTorch DataLoader or with the
    simplified train() function.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Path to CAMELS-US data directory
    basins : List[str], optional
        List of basin IDs to use. If None, load all available basins.
    forcings : List[str]
        List of forcing variables to use as input features.
        Default: CAMELS_US_FORCING_VARS
    seq_len : int
        Input sequence length (default: 365 days).
    pred_len : int
        Prediction length (default: 1 day).
    period : str
        Data period to use: 'train', 'val', 'test', or 'all'.
        If 'all', uses the entire dataset.

    Example:
        >>> # Simple training
        >>> dataset = CAMELSUSDataset("path/to/camels", basins=["01013500"])
        >>> loader = dataset.to_dataloader(batch_size=32)

        >>> # With train/val/test split
        >>> train_ds, val_ds, test_ds = dataset.split()

        >>> # Direct training with library
        >>> model, results = train(dataset, model="lstm")
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        basins: Optional[List[str]] = None,
        forcings: Optional[List[str]] = None,
        seq_len: int = 365,
        pred_len: int = 1,
        period: str = 'all',
        train_start: str = '1980-10-01',
        train_end: str = '1995-09-30',
        val_start: str = '1995-10-01',
        val_end: str = '2000-09-30',
        test_start: str = '2000-10-01',
        test_end: str = '2010-09-30',
        **kwargs
    ):
        self.forcings = forcings or CAMELS_US_FORCING_VARS
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period = period
        self.train_start = train_start
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end
        self.test_start = test_start
        self.test_end = test_end

        super().__init__(data_dir=data_dir, basins=basins, **kwargs)

    def _load_metadata(self, **kwargs):
        """Load dataset metadata."""
        self._all_basin_ids = self._discover_basins()
        # basin_ids property returns basins if specified, otherwise all discovered
        self._selected_basins = self.basins if self.basins else self._all_basin_ids

        # Pre-load all data
        self._preload_data()

    def _discover_basins(self) -> List[str]:
        """Discover available basin IDs from directory structure."""
        basin_metadata = self.data_dir / 'basin_timeseries_v1p2_metForcing_obsFlow' / \
                        'basin_dataset_public_v1p2' / 'basin_metadata' / 'gauge_information.txt'

        if basin_metadata.exists():
            with open(basin_metadata, 'r') as f:
                lines = f.readlines()[1:]
            basins = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    basin_id = parts[1]
                    basins.append(str(basin_id).zfill(8))
            return basins

        basins = []
        forcing_dir = self.data_dir / 'basin_timeseries_v1p2_metForcing_obsFlow' / \
                      'basin_dataset_public_v1p2' / 'basin_mean_forcing'

        if forcing_dir.exists():
            for forcing in ['daymet', 'maurer', 'nldas']:
                fdir = forcing_dir / forcing
                if fdir.exists():
                    for huc_dir in fdir.iterdir():
                        if huc_dir.is_dir():
                            for f in huc_dir.glob('*_lump_cida_forcing_leap.txt'):
                                basin = f.stem.split('_')[0]
                                if basin not in basins:
                                    basins.append(basin)

        return sorted(basins)

    def _preload_data(self):
        """Pre-load all basin data and build sequences."""
        self.samples = []
        self.sample_info = []  # Track basin for each sample
        self.stats = None

        print(f"Loading CAMELS-US data for {len(self.basin_ids)} basins...")

        all_data = []
        all_basins = []

        for basin_id in self.basin_ids:
            try:
                basin_data = self._load_basin(basin_id)
                df = self._basin_to_dataframe(basin_data)

                # Filter by period
                if self.period != 'all':
                    df = self._filter_by_period(df)

                if len(df) >= self.seq_len + self.pred_len:
                    all_data.append(df)
                    all_basins.append(basin_id)

            except Exception as e:
                continue

        print(f"Successfully loaded {len(all_data)} basins")

        if not all_data:
            raise ValueError("No valid basins found. Please check data directory.")

        # Concatenate and build sequences
        combined_df = pd.concat(all_data, keys=all_basins)

        # Compute statistics for normalization
        self.stats = {
            'x_mean': combined_df[self.forcings].mean().values,
            'x_std': combined_df[self.forcings].std().values,
            'y_mean': combined_df[CAMELS_US_TARGET_VAR].mean(),
            'y_std': combined_df[CAMELS_US_TARGET_VAR].std(),
        }

        # Normalize
        for i, col in enumerate(self.forcings):
            combined_df[col] = (combined_df[col] - self.stats['x_mean'][i]) / (self.stats['x_std'][i] + 1e-8)
        combined_df[CAMELS_US_TARGET_VAR] = (
            combined_df[CAMELS_US_TARGET_VAR] - self.stats['y_mean']
        ) / (self.stats['y_std'] + 1e-8)

        # Build sequences
        for (basin_id, basin_df) in combined_df.groupby(level=0):
            forcing_vals = basin_df[self.forcings].values
            target_vals = basin_df[CAMELS_US_TARGET_VAR].values

            for i in range(len(forcing_vals) - self.seq_len - self.pred_len + 1):
                x = forcing_vals[i:i + self.seq_len]
                y = target_vals[i + self.seq_len:i + self.seq_len + self.pred_len]

                if not np.isnan(x).any() and not np.isnan(y).any():
                    self.samples.append((x, y))
                    self.sample_info.append(basin_id)

        print(f"Built {len(self.samples)} samples")

    def _basin_to_dataframe(self, basin_data: BasinData) -> pd.DataFrame:
        """Convert BasinData to DataFrame."""
        dfs = []
        for var, values in basin_data.forcings.items():
            dfs.append(pd.Series(values, name=var))
        forcings_df = pd.concat(dfs, axis=1)

        if basin_data.discharge is not None:
            forcings_df[CAMELS_US_TARGET_VAR] = basin_data.discharge

        return forcings_df

    def _filter_by_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame by time period."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        if self.period == 'train':
            return df[self.train_start:self.train_end]
        elif self.period in ['val', 'validation']:
            return df[self.val_start:self.val_end]
        elif self.period == 'test':
            return df[self.test_start:self.test_end]
        return df

    def _load_basin(self, basin_id: str) -> BasinData:
        """Load data for a single basin."""
        df, area = self._load_basin_data(basin_id, self.forcings)
        geometry = None
        forcings = {var: df[var].values for var in self.forcings if var in df.columns}
        discharge = df[CAMELS_US_TARGET_VAR].values if CAMELS_US_TARGET_VAR in df.columns else None
        attributes = {}

        return BasinData(
            basin_id=basin_id,
            geometry=geometry,
            area_km2=area / 1e6,
            forcings=forcings,
            forcing_vars=self.forcings,
            discharge=discharge,
            discharge_dates=None,
            attributes=attributes
        )

    def _load_basin_data(self, basin_id: str, forcings: List[str]) -> Tuple[pd.DataFrame, float]:
        """Load combined forcing and discharge data for a basin."""
        dfs = []
        area = None

        # Try each forcing source (daymet, maurer, nldas) until one works
        forcing_sources = ['daymet', 'maurer', 'nldas']
        for source in forcing_sources:
            try:
                df_f, area = self._load_forcings(basin_id, source)
                dfs.append(df_f)
                break  # Successfully loaded, no need to try other sources
            except:
                continue

        if not dfs:
            raise ValueError(f"No forcing data for basin {basin_id}")

        df = pd.concat(dfs, axis=1)

        try:
            discharge = self._load_discharge(basin_id, area)
            df[CAMELS_US_TARGET_VAR] = discharge
            df.loc[df[CAMELS_US_TARGET_VAR] < 0, CAMELS_US_TARGET_VAR] = np.nan
        except:
            pass

        return df, area

    def _load_forcings(self, basin_id: str, forcing: str) -> Tuple[pd.DataFrame, float]:
        """Load forcing data for a basin."""
        forcing_path = self.data_dir / 'basin_timeseries_v1p2_metForcing_obsFlow' / \
                      'basin_dataset_public_v1p2' / 'basin_mean_forcing' / forcing

        file_pattern = f'**/{basin_id}_*_forcing_leap.txt'
        file_path = list(forcing_path.glob(file_pattern))

        if not file_path:
            raise FileNotFoundError(f"No forcing file for basin {basin_id}")

        file_path = file_path[0]

        with open(file_path, 'r') as f:
            f.readline()
            f.readline()
            area = int(f.readline())
            df = pd.read_csv(f, sep=r'\s+')
            df['date'] = pd.to_datetime(
                df['Year'].astype(str) + '/' + df['Mnth'].astype(str) + '/' + df['Day'].astype(str),
                format='%Y/%m/%d'
            )
            df = df.set_index('date')
            df = df.drop(['Year', 'Mnth', 'Day', 'Hr'], axis=1, errors='ignore')

        return df, area

    def _load_discharge(self, basin_id: str, area: float) -> pd.Series:
        """Load discharge data for a basin."""
        discharge_path = self.data_dir / 'basin_timeseries_v1p2_metForcing_obsFlow' / \
                        'basin_dataset_public_v1p2' / 'usgs_streamflow'

        file_pattern = f'**/{basin_id}_streamflow_qc.txt'
        file_path = list(discharge_path.glob(file_pattern))

        if not file_path:
            raise FileNotFoundError(f"No discharge file for basin {basin_id}")

        file_path = file_path[0]

        col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)
        df['date'] = pd.to_datetime(
            df['Year'].astype(str) + '/' + df['Mnth'].astype(str) + '/' + df['Day'].astype(str),
            format='%Y/%m/%d'
        )
        df = df.set_index('date')
        df['QObs'] = 28316846.592 * df['QObs'] * 86400 / (area * 1e6)

        return df['QObs']

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)

    def to_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader from this dataset.

        Parameters
        ----------
        batch_size : int
            Batch size (default: 32)
        shuffle : bool
            Whether to shuffle data (default: True)
        num_workers : int
            Number of data loading workers (default: 0)
        pin_memory : bool
            Whether to pin memory (default: True)
        drop_last : bool
            Whether to drop last incomplete batch (default: False)

        Returns
        -------
        torch.utils.data.DataLoader

        Example:
            >>> dataset = CAMELSUSDataset("path/to/camels", basins=["01013500"])
            >>> loader = dataset.to_dataloader(batch_size=32)
            >>> for x, y in loader:
            ...     print(x.shape)  # (32, 365, 7)
        """
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> Tuple['CAMELSUSDataset', 'CAMELSUSDataset', 'CAMELSUSDataset']:
        """
        Split dataset into train/val/test sets.

        Parameters
        ----------
        train_ratio : float
            Ratio of training data (default: 0.8)
        val_ratio : float
            Ratio of validation data (default: 0.1)

        Returns
        -------
        Tuple[CAMELSUSDataset, CAMELSUSDataset, CAMELSUSDataset]
            Train, validation, and test datasets

        Example:
            >>> train_ds, val_ds, test_ds = dataset.split()
            >>> train_loader = train_ds.to_dataloader()
            >>> val_loader = val_ds.to_dataloader()
        """
        n = len(self)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        indices = list(range(n))
        np.random.shuffle(indices)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_ds = CAMELSUSDataset(
            data_dir=self.data_dir,
            basins=self.basin_ids,
            forcings=self.forcings,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            **self._kwargs
        )
        train_ds.samples = [self.samples[i] for i in train_idx]
        train_ds.sample_info = [self.sample_info[i] for i in train_idx]
        train_ds.stats = self.stats

        val_ds = CAMELSUSDataset(
            data_dir=self.data_dir,
            basins=self.basin_ids,
            forcings=self.forcings,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            **self._kwargs
        )
        val_ds.samples = [self.samples[i] for i in val_idx]
        val_ds.sample_info = [self.sample_info[i] for i in val_idx]
        val_ds.stats = self.stats

        test_ds = CAMELSUSDataset(
            data_dir=self.data_dir,
            basins=self.basin_ids,
            forcings=self.forcings,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            **self._kwargs
        )
        test_ds.samples = [self.samples[i] for i in test_idx]
        test_ds.sample_info = [self.sample_info[i] for i in test_idx]
        test_ds.stats = self.stats

        return train_ds, val_ds, test_ds

    @property
    def _kwargs(self) -> Dict:
        """Get init kwargs for creating copies."""
        return {
            'period': self.period,
            'train_start': self.train_start,
            'train_end': self.train_end,
            'val_start': self.val_start,
            'val_end': self.val_end,
            'test_start': self.test_start,
            'test_end': self.test_end,
        }

    @property
    def forcing_vars(self) -> List[str]:
        return CAMELS_US_FORCING_VARS

    @property
    def target_var(self) -> str:
        return CAMELS_US_TARGET_VAR

    @property
    def basin_ids(self) -> List[str]:
        """Return list of available basin IDs."""
        return self._selected_basins

    @property
    def attribute_vars(self) -> List[str]:
        """Return list of available basin attribute names."""
        return []

    def __repr__(self) -> str:
        return (
            f"CAMELSUSDataset("
            f"basins={len(self.basin_ids)}, "
            f"samples={len(self)}, "
            f"seq_len={self.seq_len}"
            f")"
        )


def get_camels_us_basins(data_dir: Union[str, Path]) -> List[str]:
    """
    Get list of all CAMELS-US basin IDs.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Path to CAMELS-US data directory

    Returns
    -------
    List[str]
        List of 8-digit basin IDs
    """
    data_dir = Path(data_dir)
    camels = CAMELSUSDataset(data_dir=data_dir)
    return camels.basin_ids


def load_camels_us_attributes(
    data_dir: Union[str, Path],
    basins: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load catchment attributes for CAMELS-US basins.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Path to CAMELS-US data directory
    basins : Optional[List[str]]
        List of basin IDs to filter

    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame with catchment attributes
    """
    data_dir = Path(data_dir)
    camels = CAMELSUSDataset(data_dir=data_dir, basins=basins)
    basin_data = camels.get_basin(camels.basin_ids[0])

    # Load all attributes
    attribute_files = [
        'camels_clim.txt', 'camels_geol.txt', 'camels_hydro.txt',
        'camels_soil.txt', 'camels_topo.txt', 'camels_vege.txt', 'camels_name.txt'
    ]

    dfs = []
    for attr_file in attribute_files:
        file_path = data_dir / attr_file
        if file_path.exists():
            df = pd.read_csv(file_path, sep=';', header=0, dtype={'gauge_id': str})
            df = df.set_index('gauge_id')
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No attribute files found in {data_dir}")

    attributes = pd.concat(dfs, axis=1)

    if basins is not None:
        attributes = attributes.loc[attributes.index.intersection(basins)]

    return attributes
