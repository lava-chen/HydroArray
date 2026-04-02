"""
Spatial Prediction Models

This module provides models for spatial-temporal prediction tasks,
including ConvLSTM for video prediction and U-Net for image segmentation.

All models in this module are automatically registered with the global
model registry upon import.
"""

from HydroArray.ml.models.registry import register_model

# Import model classes
from .conv_lstm import (
    ConvLSTMCell,
    ConvLSTM,
    EncoderForecaster,
    SingleLayerEncoderForecaster,
)
from .unet import UNet, UNet2D

# Register models with the global registry
# This enables dynamic model creation via create_model(config)

# Register 3-layer ConvLSTM (paper configuration)
register_model("convlstm")(EncoderForecaster)

# Register single-layer ConvLSTM (lightweight version)
register_model("convlstm_single")(SingleLayerEncoderForecaster)

# Register U-Net for spatial prediction
register_model("unet")(UNet)

# Register U-Net 2D variant
register_model("unet2d")(UNet2D)

__all__ = [
    "ConvLSTMCell",
    "ConvLSTM",
    "EncoderForecaster",
    "SingleLayerEncoderForecaster",
    "UNet",
    "UNet2D",
]
