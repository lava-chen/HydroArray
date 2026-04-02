"""
Moving MNIST Dataset

A benchmark dataset for video prediction, containing sequences of
moving MNIST digits. Each sequence has 20 frames of 64x64 grayscale
images.

This dataset is registered with the global dataset registry and can be
instantiated via create_dataset().

Example:
    >>> from HydroArray.data import create_dataset
    >>> config = {'data': {'dataset': 'moving_mnist', 'data_path': 'data.npy'}}
    >>> dataset = create_dataset(config, period='train')
    >>> sample = dataset[0]
    >>> print(sample[0].shape)  # (10, 1, 64, 64) - input sequence
    >>> print(sample[1].shape)  # (10, 1, 64, 64) - target sequence
"""

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MovingMNISTDataset(Dataset):
    """
    Moving MNIST dataset for video prediction.
    
    The dataset contains sequences of moving MNIST digits. Each sample
    consists of an input sequence and a target sequence for prediction.
    
    Parameters
    ----------
    config : Any
        Configuration object containing dataset parameters
    period : str
        Dataset period ('train', 'val', or 'test')
    
    Attributes
    ----------
    data : np.ndarray
        Raw dataset array of shape (20, num_samples, 64, 64)
    seq_len : int
        Length of input sequence
    future_len : int
        Length of target sequence (prediction horizon)
    
    Example:
        >>> dataset = MovingMNISTDataset(config, period='train')
        >>> input_seq, target_seq = dataset[0]
        >>> print(input_seq.shape)   # torch.Size([10, 1, 64, 64])
        >>> print(target_seq.shape)  # torch.Size([10, 1, 64, 64])
    """
    
    def __init__(self, config: Any, period: str = 'train'):
        """
        Initialize Moving MNIST dataset.
        
        Args:
            config: Configuration object with data parameters
            period: 'train', 'val', or 'test'
        """
        super().__init__()
        
        # Extract parameters from config
        self.data_path = Path(getattr(config.data, 'data_path', None) or 
                              config.get('data', {}).get('data_path'))
        self.seq_len = getattr(config.data, 'seq_len', 
                               config.get('data', {}).get('seq_len', 10))
        self.future_len = getattr(config.data, 'future_len',
                                  config.get('data', {}).get('future_len', 10))
        train_ratio = getattr(config.data, 'train_ratio',
                              config.get('data', {}).get('train_ratio', 0.9))
        
        # Load data
        self._load_data()
        
        # Split data based on period
        self._split_data(period, train_ratio)
        
        # Normalize to [0, 1]
        self.data = self.data.astype(np.float32) / 255.0
    
    def _load_data(self):
        """Load data from file."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                f"Please download Moving MNIST dataset first."
            )
        
        self.data = np.load(self.data_path)
        # Expected shape: (20, num_samples, 64, 64)
        if self.data.ndim != 4:
            raise ValueError(
                f"Expected 4D array (frames, samples, height, width), "
                f"got shape {self.data.shape}"
            )
    
    def _split_data(self, period: str, train_ratio: float):
        """Split data into train/val sets."""
        total_samples = self.data.shape[1]
        split_idx = int(total_samples * train_ratio)
        
        if period == 'train':
            self.data = self.data[:, :split_idx, :, :]
        elif period in ['val', 'test']:
            self.data = self.data[:, split_idx:, :, :]
        else:
            raise ValueError(f"Unknown period: {period}. Use 'train', 'val', or 'test'.")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.data.shape[1]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (input_sequence, target_sequence)
            - input_sequence: torch.Tensor of shape (seq_len, 1, 64, 64)
            - target_sequence: torch.Tensor of shape (future_len, 1, 64, 64)
        """
        # Handle index wrapping
        if idx >= len(self):
            idx = idx % len(self)
        
        # Extract sequences
        input_seq = self.data[:self.seq_len, idx]  # (seq_len, 64, 64)
        target_seq = self.data[self.seq_len:self.seq_len + self.future_len, idx]
        
        # Add channel dimension and convert to tensor
        input_seq = torch.FloatTensor(input_seq).unsqueeze(1)   # (seq_len, 1, 64, 64)
        target_seq = torch.FloatTensor(target_seq).unsqueeze(1) # (future_len, 1, 64, 64)
        
        return input_seq, target_seq
    
    def get_sample_shape(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Get shape of input and target samples.
        
        Returns:
            Tuple of (input_shape, target_shape)
        """
        return (
            (self.seq_len, 1, 64, 64),
            (self.future_len, 1, 64, 64)
        )
    
    def __repr__(self) -> str:
        """String representation of dataset."""
        return (
            f"MovingMNISTDataset("
            f"samples={len(self)}, "
            f"seq_len={self.seq_len}, "
            f"future_len={self.future_len}"
            f")"
        )
