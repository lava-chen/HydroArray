"""
HydroArray - Hydrological Data Processing Library

Provides comprehensive tools for hydrological data processing, including:
    - Core data structures (HydroData, GriddedData, StationData)
    - Hydrological process calculations (runoff, evaporation)
    - Data I/O (satellite data, tables)
    - Visualization (cross-sections, spatial plots)

Example:
    >>> import HydroArray as ha
    >>>
    >>> # Hydrological calculations
    >>> result = ha.saturation_excess_runoff(data, ...)
    >>> evap = ha.three_layer_evaporation(...)
    >>>
    >>> # Data containers
    >>> gd = ha.GriddedData.from_xarray(da)
    >>> sd = ha.StationData.from_dataframe(df)
    >>>
    >>> # Visualization
    >>> ha.use_hydro_style("nature")
    >>> ha.cross_section_plot(data)
"""

__version__ = "0.1.0"

from HydroArray.core.containers import (
    HydroData,
    GriddedData,
    StationData,
    as_hydrodata,
)

# Hydrological processes - using domain layer
from HydroArray.domain.process import (
    saturation_excess_runoff,
    two_source_runoff_separation,
    three_source_runoff_separation,
    three_layer_evaporation,
    calculate_channel_section_detailed,
    calculate_cross_section_area,
)

# Data I/O - Table reader
from HydroArray.io.readers.table_reader import (
    read_hydro_table,
)

# Visualization
from HydroArray.plotting.styles import (
    use_hydro_style,
    get_colors,
    list_styles,
    C,
)

from HydroArray.plotting.export import (
    FigureExportConfig,
    configure_vector_text,
    save_figure,
    save_svg_png,
)

from HydroArray.plotting.crosssection import (
    cross_section_area_plot,
    cross_section_plot,
)

from HydroArray.plotting.hydrology import (
    plot_hydrograph,
    plot_precip_hydrograph,
    plot_event_panel,
    plot_flow_duration_curve,
    plot_metric_bars,
)

from HydroArray.plotting.spatial import (
    satellite_plot,
    satellite_animation,
    get_available_times,
)

# Utilities
from HydroArray.utils.file_parser import (
    parse_filename,
    parse_folder,
    parse_path,
    batch_parse_files,
    get_datetime_range,
    get_unique_sources,
)

from HydroArray.utils.rounding import (
    round_to_n_sig_figs,
    round_area,
    round_distance,
    round_width,
)

from HydroArray.utils.metrics import (
    calculate_nse,
    calculate_rmse,
    calculate_kge,
    calculate_pbias,
    calculate_bias,
    calculate_mae,
    calculate_r_squared,
    calculate_nse_log,
    mm_to_cms,
    cms_to_mm,
    evaluate_model,
)

from HydroArray.utils.diurnal import (
    diurnal_factor,
    hourly_pet_from_daily,
)

# Analysis
from HydroArray.analysis import (
    volume_error_pct,
    peak_error_pct,
    peak_time_lag,
    nse,
    rmse,
    kge,
    pbias,
    r_squared,
    regression_summary,
)

# Traditional hydrological models
from HydroArray.domain.models import (
    ModelParameters,
    WaterBalanceModel,
    RoutingModel,
    HydrologyModel,
    WaterBalanceRouted,
    GridNode,
    XinAnjiangModel,
    XinAnjiangParameters,
    HyMODModel,
    HyMODParameters,
    SACModel,
    SACParameters,
    CRESTModel,
    CRESTParameters,
    create_model,
    create_water_balance_model,
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

# Model configuration
from HydroArray.config import (
    TaskConfig,
    ModelConfig,
    ModelType,
    RoutingType,
    ParametersManager,
)

# Calibration
from HydroArray.domain.calibration import (
    CalibrationConfig,
    CalibrationResult as _LegacyCalibrationResult,
    ParameterBounds,
    GeneticAlgorithm,
    SCEOptimizer,
    ScipyOptimizer,
)
# 老 calibrate 重命名为 calibrate_legacy,避免和新 facade 撞名
from HydroArray.domain.calibration import calibrate as calibrate_legacy

# High-level auto-calibration (ref-table driven, multi-start, normalised)
from HydroArray.calibration import (
    auto_calibrate,
    AutoCalibrateResult,
    CalibrationResult,
    Calibrator,
    CalibratorConfig,
    CalibrationPipeline,
    CalibratorPreset,
    list_presets,
    get_preset,
    make_calibrator_from_preset,
    resume_from_result,
    resume_from_calibrator_checkpoint,
)
# 顶层 ha.calibrate 指向新 facade(覆盖老的)
from HydroArray.calibration._facade import calibrate

# Data Assimilation
from HydroArray.domain.assimilation import (
    AssimilationResult,
    DataAssimilator,
    EnsembleKalmanFilter,
    ParticleFilter,
    OfflineDA,
    create_assimilator,
)

# Sensitivity Analysis
from HydroArray.domain.sensitivity import (
    SensitivityResult,
    SobolAnalyzer,
    MorrisAnalyzer,
    ParameterRelativeSensitivity,
    analyze_sensitivity,
)

# Ensemble Learning
from HydroArray.domain.ensemble import (
    EnsembleMember,
    EnsembleResult,
    ModelEnsemble,
    WeightedModelEnsemble,
    ParameterEnsemble,
    create_structural_ensemble,
)


__all__ = [
    "__version__",

    # Core data structures
    "HydroData",
    "GriddedData",
    "StationData",
    "as_hydrodata",

    # Hydrological processes
    "saturation_excess_runoff",
    "two_source_runoff_separation",
    "three_source_runoff_separation",
    "three_layer_evaporation",
    "calculate_channel_section_detailed",
    "calculate_cross_section_area",

    # Data I/O
    "read_hydro_table",

    # Visualization
    "use_hydro_style",
    "get_colors",
    "list_styles",
    "C",
    "FigureExportConfig",
    "configure_vector_text",
    "save_figure",
    "save_svg_png",
    "cross_section_area_plot",
    "cross_section_plot",
    "plot_hydrograph",
    "plot_precip_hydrograph",
    "plot_event_panel",
    "plot_flow_duration_curve",
    "plot_metric_bars",
    "satellite_plot",
    "satellite_animation",
    "get_available_times",

    # Utilities
    "parse_filename",
    "parse_folder",
    "parse_path",
    "batch_parse_files",
    "get_datetime_range",
    "get_unique_sources",
    "round_to_n_sig_figs",
    "round_area",
    "round_distance",
    "round_width",

    # Traditional models - base classes
    "ModelParameters",
    "WaterBalanceModel",
    "RoutingModel",
    "HydrologyModel",
    "WaterBalanceRouted",
    "GridNode",

    # Traditional models - implementations
    "XinAnjiangModel",
    "XinAnjiangParameters",
    "HyMODModel",
    "HyMODParameters",
    "SACModel",
    "SACParameters",
    "CRESTModel",
    "CRESTParameters",

    # Model factory
    "create_model",
    "create_water_balance_model",
    "create_hydrology_model",
    "get_available_models",

    # Routing models
    "LinearRouting",
    "LinearRoutingParameters",
    "KinematicRouting",
    "KinematicRoutingParameters",
    "MuskingumRouting",
    "MuskingumParameters",

    # Model configuration
    "TaskConfig",
    "ModelConfig",
    "ModelType",
    "RoutingType",
    "ParametersManager",

    # Metrics
    "calculate_nse",
    "calculate_rmse",
    "calculate_kge",
    "calculate_pbias",
    "calculate_bias",
    "calculate_mae",
    "calculate_r_squared",
    "calculate_nse_log",
    "mm_to_cms",
    "cms_to_mm",
    "evaluate_model",
    # Diurnal downscaling
    "diurnal_factor",
    "hourly_pet_from_daily",
    # Analysis (model-agnostic hydrology metrics)
    "volume_error_pct",
    "peak_error_pct",
    "peak_time_lag",
    "nse",
    "rmse",
    "kge",
    "pbias",
    "r_squared",
    "regression_summary",

    # Calibration
    "CalibrationConfig",
    "CalibrationResult",
    "ParameterBounds",
    "GeneticAlgorithm",
    "SCEOptimizer",
    "ScipyOptimizer",
    "calibrate_legacy",
    "calibrate",
    "auto_calibrate",
    "AutoCalibrateResult",
    "Calibrator",
    "CalibratorConfig",
    "CalibrationPipeline",
    "CalibratorPreset",
    "list_presets",
    "get_preset",
    "make_calibrator_from_preset",
    "resume_from_result",
    "resume_from_calibrator_checkpoint",

    # Data Assimilation
    "AssimilationResult",
    "DataAssimilator",
    "EnsembleKalmanFilter",
    "ParticleFilter",
    "OfflineDA",
    "create_assimilator",

    # Sensitivity Analysis
    "SensitivityResult",
    "SobolAnalyzer",
    "MorrisAnalyzer",
    "ParameterRelativeSensitivity",
    "analyze_sensitivity",

    # Ensemble Learning
    "EnsembleMember",
    "EnsembleResult",
    "ModelEnsemble",
    "WeightedModelEnsemble",
    "ParameterEnsemble",
    "create_structural_ensemble",
]


def __dir__():
    return sorted(__all__)
