"""
Machine Learning Models Module

This module provides a comprehensive collection of machine learning models
for hydrological and spatiotemporal prediction tasks. All models are
registered with the model registry for dynamic instantiation.

Available Model Categories:
    - Spatial Models: ConvLSTM, U-Net for spatial-temporal prediction
    - Sequence Models: LSTM, GRU, Transformer for time series
    - Generative Models: Diffusion, VAE for data generation
    - Hydrological Models: Rainfall-runoff, flood forecasting

Example:
    >>> from HydroArray.ml.models import create_model, list_available_models
    >>> 
    >>> # List all available models
    >>> models = list_available_models()
    >>> print(models)
    ['convlstm', 'unet']
    
    >>> # Create model from configuration
    >>> model = create_model(config)
    
    >>> # Or create by name
    >>> from HydroArray.ml.models import get_model_class
    >>> ConvLSTM = get_model_class("convlstm")
    >>> model = ConvLSTM(config)

Model Registration:
    To add a new model, use the @register_model decorator:
    
    >>> from HydroArray.ml.models import register_model
    >>> 
    >>> @register_model("my_model")
    >>> class MyModel(nn.Module):
    ...     def __init__(self, config):
    ...         super().__init__()
    ...         # Implementation
"""

# Import and re-export registry functions
from .registry import (
    # Core registry class
    ModelRegistry,
    MODEL_REGISTRY,
    
    # Registration decorator
    register_model,
    
    # Factory functions
    create_model,
    get_model_class,
    list_available_models,
    is_model_available,
)

# Import spatial models
from .spatial import (
    ConvLSTMCell,
    ConvLSTM,
    EncoderForecaster,
    SingleLayerEncoderForecaster,
    UNet,
    UNet2D,
)

# Import generative models
from .generative import (
    DiffusionModel,
)

# Import sequence models (automatically registers lstm, bilstm)
from . import sequence

# Import hydrological models (placeholders - implementation pending)
# from .flood_forecast import FloodForecastModel
# from .rainfall_runoff import RainfallRunoffModel
# from .water_quality import WaterQualityModel

# Model registration happens automatically when modules are imported
# The following models are registered:
# - "convlstm": EncoderForecaster (3-layer ConvLSTM)
# - "convlstm_single": SingleLayerEncoderForecaster (1-layer ConvLSTM)
# - "unet": UNet
# - "unet2d": UNet2D
# - "lstm": SimpleLSTM
# - "bilstm": BidirectionalLSTM
# - "diffusion": DiffusionModel

__all__ = [
    # Registry functions
    "ModelRegistry",
    "MODEL_REGISTRY",
    "register_model",
    "create_model",
    "get_model_class",
    "list_available_models",
    "is_model_available",
    
    # Spatial models
    "ConvLSTMCell",
    "ConvLSTM",
    "EncoderForecaster",
    "SingleLayerEncoderForecaster",
    "UNet",
    "UNet2D",
    
    # Sequence models
    "SimpleLSTM",
    "BidirectionalLSTM",
    
    # Generative models
    "DiffusionModel",
    
    # Hydrological models (pending implementation)
    # "FloodForecastModel",
    # "RainfallRunoffModel",
    # "WaterQualityModel",
]
