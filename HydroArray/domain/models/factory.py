"""
Hydrological Model Factory

Factory functions for creating and configuring hydrological models.
"""

from typing import Optional, Dict, Any

from HydroArray.domain.models.base import (
    WaterBalanceModel,
    RoutingModel,
    HydrologyModel,
    WaterBalanceRouted,
)
from HydroArray.domain.models.xinanjiang import XinAnjiangModel, XinAnjiangParameters
from HydroArray.domain.models.hymod import HyMODModel, HyMODParameters
from HydroArray.domain.models.sac import SACModel, SACParameters
from HydroArray.domain.models.crest import CRESTModel, CRESTParameters
from HydroArray.domain.routing import (
    LinearRouting,
    LinearRoutingParameters,
    KinematicRouting,
    KinematicRoutingParameters,
    MuskingumRouting,
    MuskingumParameters,
)
from HydroArray.config.parameters import get_parameters_class, ParametersManager
from HydroArray.config.task import ModelConfig, ModelType, RoutingType


# Registry for water balance models
_WB_MODEL_REGISTRY: Dict[str, type] = {
    'xinanjiang': XinAnjiangModel,
    'hymod': HyMODModel,
    'sac': SACModel,
    'crest': CRESTModel,
}

# Registry for routing models
_ROUTING_MODEL_REGISTRY: Dict[str, type] = {
    'linear': LinearRouting,
    'kinematic': KinematicRouting,
    'muskingum': MuskingumRouting,
}


def create_water_balance_model(model_name: str) -> WaterBalanceModel:
    """Create a water balance model by name.

    Args:
        model_name: Name of the model ('xinanjiang', 'hymod', 'sac', 'crest').

    Returns:
        Instance of the requested model.

    Raises:
        ValueError: If model_name is not recognized.
    """
    if model_name.lower() not in _WB_MODEL_REGISTRY:
        available = list(_WB_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    model_class = _WB_MODEL_REGISTRY[model_name.lower()]
    return model_class()


def create_routing_model(routing_type: RoutingType) -> Optional[RoutingModel]:
    """Create a routing model by type.

    Args:
        routing_type: Type of routing model.

    Returns:
        Instance of the routing model, or None if routing_type is NONE.
    """
    if routing_type == RoutingType.NONE:
        return None

    routing_name = routing_type.value
    if routing_name not in _ROUTING_MODEL_REGISTRY:
        return None

    routing_class = _ROUTING_MODEL_REGISTRY[routing_name]
    return routing_class()


def create_hydrology_model(config: ModelConfig) -> HydrologyModel:
    """Create a complete hydrology model from configuration.

    Args:
        config: Model configuration containing model type and parameters.

    Returns:
        Configured hydrology model (WaterBalanceRouted or similar).

    Raises:
        ValueError: If model type is not supported.
    """
    wb_model = create_water_balance_model(config.water_balance or config.model_type.value)
    routing = create_routing_model(config.routing or RoutingType.NONE)

    model = WaterBalanceRouted(water_balance=wb_model, routing=routing)

    # Initialize with parameters if provided
    if config.parameters:
        param_class = get_parameters_class(config.model_type.value)
        if param_class:
            params = param_class.from_dict(config.parameters)
            # Need nodes for initialization - will be done at runtime
            model._params = params

    return model


def get_available_models() -> Dict[str, list]:
    """Get all available model types.

    Returns:
        Dictionary with 'water_balance' and 'routing' model lists.
    """
    return {
        'water_balance': list(_WB_MODEL_REGISTRY.keys()),
        'routing': [rt.value for rt in RoutingType if rt != RoutingType.NONE],
    }


# Alias for create_model compatible with ML module
def create_model(config: ModelConfig) -> HydrologyModel:
    """Create a hydrology model from configuration.

    This is an alias for create_hydrology_model to maintain
    API consistency with the ML model creation.
    """
    return create_hydrology_model(config)
