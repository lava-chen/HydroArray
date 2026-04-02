"""
Sequence Prediction Models

This module provides models for time series and sequence prediction tasks,
including LSTM, GRU, and Transformer architectures.

All models are automatically registered with the global model registry.
"""

from HydroArray.ml.models.registry import register_model

# Import LSTM models
from .lstm import SimpleLSTM, BidirectionalLSTM

# Register models
register_model("lstm")(SimpleLSTM)
register_model("bilstm")(BidirectionalLSTM)

__all__ = [
    "SimpleLSTM",
    "BidirectionalLSTM",
]
