"""
Machine Learning Module for HydroArray

This module provides simple training interfaces for hydrological ML models.

Quick Start:
    >>> from HydroArray.datasets import CAMELSUSDataset
    >>> from HydroArray.ml import train

    >>> # Train in 3 lines
    >>> dataset = CAMELSUSDataset("path/to/camels", basins=["01013500"])
    >>> model, results = train(dataset, model="lstm")

    >>> # Or with more control
    >>> model, results = train(
    ...     dataset=dataset,
    ...     model="lstm",
    ...     hidden_dim=128,
    ...     epochs=50
    ... )
"""

from . import models
from . import trainer
from .trainer import (
    BaseTrainer,
    SequenceTrainer,
    SpatialTrainer,
    create_trainer
)

# Import simple training interface
from HydroArray.ml.train import train

__all__ = [
    # Submodules
    "models",
    "trainer",
    # Trainer classes
    "BaseTrainer",
    "SequenceTrainer",
    "SpatialTrainer",
    "create_trainer",
    # Simple training interface
    "train",
]
