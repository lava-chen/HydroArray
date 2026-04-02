"""
Hydrological Models Module

Contains traditional hydrological water balance and routing models.
"""

from HydroArray.domain.models.base import (
    ModelParameters,
    WaterBalanceModel,
    RoutingModel,
    HydrologyModel,
    WaterBalanceRouted,
    GridNode,
)

from HydroArray.domain.models.xinanjiang import (
    XinAnjiangModel,
    XinAnjiangParameters,
)

from HydroArray.domain.models.hymod import (
    HyMODModel,
    HyMODParameters,
)

from HydroArray.domain.models.sac import (
    SACModel,
    SACParameters,
)

from HydroArray.domain.models.crest import (
    CRESTModel,
    CRESTParameters,
)

from HydroArray.domain.models.factory import (
    create_model,
    create_water_balance_model,
    create_routing_model,
    create_hydrology_model,
    get_available_models,
)

__all__ = [
    # Base classes
    "ModelParameters",
    "WaterBalanceModel",
    "RoutingModel",
    "HydrologyModel",
    "WaterBalanceRouted",
    "GridNode",
    # XinAnjiang model
    "XinAnjiangModel",
    "XinAnjiangParameters",
    # HyMOD model
    "HyMODModel",
    "HyMODParameters",
    # SAC-SMA model
    "SACModel",
    "SACParameters",
    # CREST model
    "CRESTModel",
    "CRESTParameters",
    # Factory functions
    "create_model",
    "create_water_balance_model",
    "create_routing_model",
    "create_hydrology_model",
    "get_available_models",
]
