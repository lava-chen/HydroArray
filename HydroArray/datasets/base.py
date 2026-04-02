"""
Base Dataset Classes for HydroArray

Provides unified dataset interfaces that bridge HydroArray's core data
structures (HydroData) with PyTorch DataLoader for ML training.

This module ensures seamless integration between:
    - HydroArray.core data containers (HydroData)
    - PyTorch Dataset and DataLoader
    - ML model training pipelines

Example:
    >>> from HydroArray.datasets import HydroDataset, create_dataloader
    >>> from HydroArray.core import HydroData

    >>> # Create from HydroData
    >>> hydro_data = HydroData.from_xarray(xr_dataset)
    >>> dataset = HydroDataset.from_hydrodata(
    ...     hydro_data,
    ...     input_vars=['rainfall', 'temperature'],
    ...     target_vars=['discharge'],
    ...     seq_len=10
    ... )

    >>> # Create DataLoader
    >>> loader = create_dataloader(dataset, batch_size=32, shuffle=True)
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from HydroArray.core.containers import HydroData
    HAS_CORE = True
except ImportError:
    HAS_CORE = False
    HydroData = None


def _build_sequences(data: np.ndarray, seq_len: int, pred_len: int):
    """Build sequences using sliding window."""
    total_len = seq_len + pred_len
    n_samples = len(data) - total_len + 1

    if n_samples <= 0:
        return np.array([]), np.array([])

    samples = np.array([data[i:i + seq_len] for i in range(n_samples)])
    targets = np.array([data[i + seq_len:i + total_len] for i in range(n_samples)])

    return samples, targets


class HydroDataset(Dataset):
    """
    Base dataset class for HydroArray data.

    This class provides a bridge between HydroArray's core data structures
    and PyTorch's Dataset interface. It handles:
        - Variable selection from HydroData
        - Sequence windowing for time series
        - Train/val/test splitting
        - Data normalization

    Parameters
    ----------
    data : Union[HydroData, Dict[str, np.ndarray], np.ndarray]
        Input data in HydroData format or numpy arrays
    input_vars : List[str]
        List of input variable names
    target_vars : List[str]
        List of target variable names
    seq_len : int
        Length of input sequences (lookback window)
    pred_len : int
        Length of prediction sequences (forecast horizon)
    transform : Optional[Callable]
        Optional transform function for data augmentation

    Attributes
    ----------
    inputs : np.ndarray
        Input data array of shape (num_samples, seq_len, num_features)
    targets : np.ndarray
        Target data array of shape (num_samples, pred_len, num_targets)

    Example:
        >>> # From HydroData
        >>> hydro_data = HydroData.from_xarray(dataset)
        >>> dataset = HydroDataset(
        ...     data=hydro_data,
        ...     input_vars=['rainfall', 'temp'],
        ...     target_vars=['discharge'],
        ...     seq_len=10,
        ...     pred_len=5
        ... )

        >>> # Access sample
        >>> x, y = dataset[0]
        >>> print(x.shape)  # (10, 2) - 10 timesteps, 2 features
        >>> print(y.shape)  # (5, 1) - 5 timesteps, 1 target
    """

    def __init__(
        self,
        data: Union[Any, Dict[str, np.ndarray], np.ndarray],
        input_vars: List[str],
        target_vars: List[str],
        seq_len: int = 10,
        pred_len: int = 1,
        transform: Optional[Callable] = None
    ):
        super().__init__()

        self.input_vars = input_vars
        self.target_vars = target_vars
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.transform = transform

        # Extract data from various formats
        self._extract_data(data)

        # Build sequences
        self._build_sequences()

        # Validate data
        self._validate()

    def _extract_data(self, data: Union[Any, Dict, np.ndarray]):
        """Extract numpy arrays from various data formats."""
        if HAS_CORE and isinstance(data, HydroData):
            # From HydroData container
            if isinstance(data.data, dict):
                self.input_data = np.stack([
                    data.data[var] for var in self.input_vars
                ], axis=-1)  # (time, features)
                self.target_data = np.stack([
                    data.data[var] for var in self.target_vars
                ], axis=-1)  # (time, targets)
            else:
                raise ValueError("HydroData with dict data required for variable selection")
        elif isinstance(data, dict):
            # From dictionary
            self.input_data = np.stack([
                data[var] for var in self.input_vars
            ], axis=-1)
            self.target_data = np.stack([
                data[var] for var in self.target_vars
            ], axis=-1)
        elif isinstance(data, np.ndarray):
            # From single numpy array (assume all are inputs)
            if data.ndim == 2:
                self.input_data = data
                self.target_data = data[:, :len(self.target_vars)]
            else:
                raise ValueError(f"Unsupported data shape: {data.shape}")
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _build_sequences(self):
        """Build input-target sequences using sliding window."""
        total_len = self.seq_len + self.pred_len
        num_samples = len(self.input_data) - total_len + 1

        if num_samples <= 0:
            raise ValueError(
                f"Not enough data for seq_len={self.seq_len} and "
                f"pred_len={self.pred_len}. Data length: {len(self.input_data)}"
            )

        self.inputs = []
        self.targets = []

        for i in range(num_samples):
            # Input sequence
            x = self.input_data[i:i + self.seq_len]
            # Target sequence
            y = self.target_data[i + self.seq_len:i + self.seq_len + self.pred_len]

            self.inputs.append(x)
            self.targets.append(y)

        self.inputs = np.array(self.inputs)  # (num_samples, seq_len, num_features)
        self.targets = np.array(self.targets)  # (num_samples, pred_len, num_targets)

    def _validate(self):
        """Validate dataset structure."""
        assert self.inputs.shape[0] == self.targets.shape[0], \
            "Input and target sample counts don't match"
        assert self.inputs.shape[1] == self.seq_len, \
            f"Input sequence length mismatch: {self.inputs.shape[1]} vs {self.seq_len}"
        assert self.targets.shape[1] == self.pred_len, \
            f"Target sequence length mismatch: {self.targets.shape[1]} vs {self.pred_len}"

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (input_sequence, target_sequence)
        """
        x = torch.FloatTensor(self.inputs[idx])
        y = torch.FloatTensor(self.targets[idx])

        if self.transform:
            x = self.transform(x)

        return x, y

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for input and target variables.

        Returns
        -------
        Dict with mean, std, min, max for each variable
        """
        stats = {
            'inputs': {},
            'targets': {}
        }

        # Input stats
        for i, var in enumerate(self.input_vars):
            var_data = self.inputs[:, :, i].flatten()
            stats['inputs'][var] = {
                'mean': float(np.mean(var_data)),
                'std': float(np.std(var_data)),
                'min': float(np.min(var_data)),
                'max': float(np.max(var_data))
            }

        # Target stats
        for i, var in enumerate(self.target_vars):
            var_data = self.targets[:, :, i].flatten()
            stats['targets'][var] = {
                'mean': float(np.mean(var_data)),
                'std': float(np.std(var_data)),
                'min': float(np.min(var_data)),
                'max': float(np.max(var_data))
            }

        return stats

    def normalize(self, method: str = 'standard') -> 'HydroDataset':
        """
        Normalize dataset.

        Parameters
        ----------
        method : str
            'standard' (zero mean, unit std) or 'minmax'

        Returns
        -------
        self for method chaining
        """
        if method == 'standard':
            self.inputs = (self.inputs - np.mean(self.inputs, axis=0)) / \
                         (np.std(self.inputs, axis=0) + 1e-8)
            self.targets = (self.targets - np.mean(self.targets, axis=0)) / \
                          (np.std(self.targets, axis=0) + 1e-8)
        elif method == 'minmax':
            self.inputs = (self.inputs - np.min(self.inputs, axis=0)) / \
                         (np.max(self.inputs, axis=0) - np.min(self.inputs, axis=0) + 1e-8)
            self.targets = (self.targets - np.min(self.targets, axis=0)) / \
                          (np.max(self.targets, axis=0) - np.min(self.targets, axis=0) + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return self

    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """
        Split dataset into train/val/test sets.

        Parameters
        ----------
        train_ratio : float
            Ratio of training data
        val_ratio : float
            Ratio of validation data

        Returns
        -------
        Tuple of (train_dataset, val_dataset, test_dataset)
        """
        n = len(self)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        return (
            Subset(self, train_idx),
            Subset(self, val_idx),
            Subset(self, test_idx)
        )

    @classmethod
    def from_numpy(
        cls,
        data: np.ndarray,
        seq_len: int = 10,
        pred_len: int = 1,
        normalize: bool = False
    ):
        """
        Create dataset from numpy array.

        Parameters
        ----------
        data : np.ndarray
            Input array of shape (time, features) or (time,)
        seq_len : int
            Input sequence length
        pred_len : int
            Prediction sequence length
        normalize : bool
            Whether to normalize the data

        Returns
        -------
        HydroDataset
        """
        samples, targets = _build_sequences(data, seq_len, pred_len)
        dataset = cls.__new__(cls)
        dataset.input_vars = ['var_0']
        dataset.target_vars = ['var_0']
        dataset.seq_len = seq_len
        dataset.pred_len = pred_len
        dataset.transform = None
        dataset.inputs = samples
        dataset.targets = targets

        if normalize:
            dataset.normalize()

        return dataset

    @classmethod
    def from_dataframe(
        cls,
        df,
        input_vars: List[str],
        target_vars: List[str],
        seq_len: int = 10,
        pred_len: int = 1,
        normalize: bool = False
    ):
        """
        Create dataset from pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with columns for input and target variables
        input_vars : List[str]
            List of input variable names
        target_vars : List[str]
            List of target variable names
        seq_len : int
            Input sequence length
        pred_len : int
            Prediction sequence length
        normalize : bool
            Whether to normalize the data

        Returns
        -------
        HydroDataset
        """
        input_data = df[input_vars].values
        target_data = df[target_vars].values

        X, _ = _build_sequences(input_data, seq_len, pred_len)
        _, y = _build_sequences(target_data, seq_len, pred_len)

        dataset = cls.__new__(cls)
        dataset.input_vars = input_vars
        dataset.target_vars = target_vars
        dataset.seq_len = seq_len
        dataset.pred_len = pred_len
        dataset.transform = None
        dataset.inputs = X
        dataset.targets = y

        if normalize:
            dataset.normalize()

        return dataset

    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
        input_vars: Optional[List[str]] = None,
        target_vars: Optional[List[str]] = None,
        seq_len: int = 10,
        pred_len: int = 1,
        normalize: bool = False
    ):
        """
        Load dataset from file (npy, csv, nc).

        Parameters
        ----------
        filepath : str or Path
            Path to the data file
        input_vars : List[str], optional
            Input variable names (required for CSV)
        target_vars : List[str], optional
            Target variable names (required for CSV)
        seq_len : int
            Input sequence length
        pred_len : int
            Prediction sequence length
        normalize : bool
            Whether to normalize the data

        Returns
        -------
        HydroDataset
        """
        filepath = Path(filepath)

        if filepath.suffix == '.npy':
            data = np.load(filepath)
            return cls.from_numpy(data, seq_len, pred_len, normalize)

        elif filepath.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(filepath)
            if input_vars is None or target_vars is None:
                raise ValueError("input_vars and target_vars required for CSV files")
            return cls.from_dataframe(df, input_vars, target_vars, seq_len, pred_len, normalize)

        else:
            raise ValueError(f"Unsupported format: {filepath.suffix}")

    @classmethod
    def from_hydrodata(
        cls,
        hydro_data: Any,
        input_vars: List[str],
        target_vars: List[str],
        seq_len: int = 10,
        pred_len: int = 1,
        transform: Optional[Callable] = None
    ) -> 'HydroDataset':
        """
        Create dataset from HydroData container.

        Parameters
        ----------
        hydro_data : HydroData
            HydroArray core data container
        input_vars : List[str]
            Input variable names
        target_vars : List[str]
            Target variable names
        seq_len : int
            Input sequence length
        pred_len : int
            Prediction sequence length
        transform : Optional[Callable]
            Optional transform function

        Returns
        -------
        HydroDataset
        """
        if not HAS_CORE:
            raise ImportError("HydroArray core module not available")

        return cls(
            data=hydro_data,
            input_vars=input_vars,
            target_vars=target_vars,
            seq_len=seq_len,
            pred_len=pred_len,
            transform=transform
        )

    def __repr__(self) -> str:
        return (
            f"HydroDataset("
            f"samples={len(self)}, "
            f"seq_len={self.seq_len}, "
            f"pred_len={self.pred_len}, "
            f"inputs={self.input_vars}, "
            f"targets={self.target_vars}"
            f")"
        )


class Subset(Dataset):
    """Subset of HydroDataset for train/val/test splitting."""

    def __init__(self, dataset: HydroDataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[self.indices[idx]]


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create PyTorch DataLoader with HydroArray defaults.

    Parameters
    ----------
    dataset : Dataset
        PyTorch Dataset instance
    batch_size : int
        Batch size
    shuffle : bool
        Whether to shuffle data
    num_workers : int
        Number of data loading workers
    pin_memory : bool
        Whether to pin memory for GPU transfer
    drop_last : bool
        Whether to drop last incomplete batch

    Returns
    -------
    DataLoader

    Example:
        >>> dataset = HydroDataset(...)
        >>> loader = create_dataloader(dataset, batch_size=32, shuffle=True)
        >>> for x, y in loader:
        ...     print(x.shape)  # (32, 10, 2)
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


class DataTransformer:
    """
    Data transformation utilities for HydroDataset.

    Provides common transformations like normalization,
    log transform, and masking.

    Example:
        >>> transform = DataTransformer()
        >>> transform.add_normalize(mean=0.5, std=0.25)
        >>> transform.add_log_transform()

        >>> dataset = HydroDataset(..., transform=transform)
    """

    def __init__(self):
        self.transforms = []

    def add_normalize(self, mean: float = 0.0, std: float = 1.0):
        """Add standard normalization."""
        self.transforms.append(lambda x: (x - mean) / (std + 1e-8))
        return self

    def add_minmax_scale(self, min_val: float = 0.0, max_val: float = 1.0):
        """Add min-max scaling."""
        self.transforms.append(
            lambda x: (x - min_val) / (max_val - min_val + 1e-8)
        )
        return self

    def add_log_transform(self, offset: float = 1e-6):
        """Add log transform (useful for skewed data like rainfall)."""
        self.transforms.append(lambda x: torch.log(x + offset))
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all transforms."""
        for transform in self.transforms:
            x = transform(x)
        return x
