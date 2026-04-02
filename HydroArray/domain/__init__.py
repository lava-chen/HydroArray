"""
HydroArray Domain Module

Hydrological domain concepts organized by responsibility:

- process: Hydrological processes (runoff, evaporation, routing)
- observation: Observed hydrological data (discharge, water level)
- forcing: Meteorological forcing data (precipitation, temperature)
- basin: Basin representation and relationships

The domain layer encapsulates the core hydrological logic, independent
of data I/O or ML training concerns.

Example:
    >>> from HydroArray.domain.process import saturation_excess_runoff
    >>> from HydroArray.domain.observation import Discharge
"""

from HydroArray.domain.process import (
    saturation_excess_runoff,
    two_source_runoff_separation,
    three_source_runoff_separation,
    three_layer_evaporation,
    calculate_channel_section_detailed,
    calculate_cross_section_area,
)

from HydroArray.domain.observation import (
    to_daily,
    frequency,
)

from HydroArray.domain.models import (
    ModelParameters,
    WaterBalanceModel,
    RoutingModel,
    HydrologyModel,
    WaterBalanceRouted,
    GridNode,
    # Concrete models
    XinAnjiangModel,
    XinAnjiangParameters,
    HyMODModel,
    HyMODParameters,
    SACModel,
    SACParameters,
    CRESTModel,
    CRESTParameters,
    # Factory
    create_model,
    create_water_balance_model,
    create_routing_model,
    create_hydrology_model,
    get_available_models,
)

from HydroArray.domain.routing import (
    LinearRouting,
    LinearRoutingParameters,
    KinematicRouting,
    KinematicRoutingParameters,
    MuskingumRouting,
    MuskingumParameters,
)

from HydroArray.domain.calibration import (
    CalibrationConfig,
    CalibrationResult,
    ParameterBounds,
    GeneticAlgorithm,
    SCEOptimizer,
    ScipyOptimizer,
    calibrate,
)

from HydroArray.domain.assimilation import (
    AssimilationResult,
    DataAssimilator,
    EnsembleKalmanFilter,
    ParticleFilter,
    OfflineDA,
    create_assimilator,
)

from HydroArray.domain.sensitivity import (
    SensitivityResult,
    SensitivityAnalyzer,
    SobolAnalyzer,
    MorrisAnalyzer,
    ParameterRelativeSensitivity,
    analyze_sensitivity,
)

from HydroArray.domain.ensemble import (
    EnsembleMember,
    EnsembleResult,
    ModelEnsemble,
    WeightedModelEnsemble,
    ParameterEnsemble,
    create_structural_ensemble,
)

__all__ = [
    # Process
    "saturation_excess_runoff",
    "two_source_runoff_separation",
    "three_source_runoff_separation",
    "three_layer_evaporation",
    "calculate_channel_section",
    "calculate_cross_section_area",
    # Observation
    "to_daily",
    "frequency",
    # Base classes
    "ModelParameters",
    "WaterBalanceModel",
    "RoutingModel",
    "HydrologyModel",
    "WaterBalanceRouted",
    "GridNode",
    # XinAnjiang
    "XinAnjiangModel",
    "XinAnjiangParameters",
    # HyMOD
    "HyMODModel",
    "HyMODParameters",
    # SAC-SMA
    "SACModel",
    "SACParameters",
    # CREST
    "CRESTModel",
    "CRESTParameters",
    # Factory functions
    "create_model",
    "create_water_balance_model",
    "create_routing_model",
    "create_hydrology_model",
    "get_available_models",
    # Routing
    "LinearRouting",
    "LinearRoutingParameters",
    "KinematicRouting",
    "KinematicRoutingParameters",
    "MuskingumRouting",
    "MuskingumParameters",
    # Calibration
    "CalibrationConfig",
    "CalibrationResult",
    "ParameterBounds",
    "GeneticAlgorithm",
    "SCEOptimizer",
    "ScipyOptimizer",
    "calibrate",
    # Data Assimilation
    "AssimilationResult",
    "DataAssimilator",
    "EnsembleKalmanFilter",
    "ParticleFilter",
    "OfflineDA",
    "create_assimilator",
    # Sensitivity Analysis
    "SensitivityResult",
    "SensitivityAnalyzer",
    "SobolAnalyzer",
    "MorrisAnalyzer",
    "ParameterRelativeSensitivity",
    "analyze_sensitivity",
    # Ensemble
    "EnsembleMember",
    "EnsembleResult",
    "ModelEnsemble",
    "WeightedModelEnsemble",
    "ParameterEnsemble",
    "create_structural_ensemble",
]
