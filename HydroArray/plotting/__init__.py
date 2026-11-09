"""
可视化模块
"""

from HydroArray.plotting.styles import use_hydro_style, get_colors, list_styles
from HydroArray.plotting.export import (
    FigureExportConfig,
    configure_vector_text,
    save_figure,
    save_svg_png,
)

from HydroArray.plotting.crosssection import (
    cross_section_plot,
    cross_section_area_plot,
)

from HydroArray.plotting.timeseries import (
    plot_loss_curve,
    plot_predictions_vs_observations,
    plot_time_series_comparison,
    plot_metrics_summary,
)

from HydroArray.plotting.hydrology import (
    plot_hydrograph,
    plot_precip_hydrograph,
    plot_event_panel,
    plot_flow_duration_curve,
    plot_metric_bars,
)

__all__ = [
    # Styles
    "use_hydro_style",
    "get_colors",
    "list_styles",
    "FigureExportConfig",
    "configure_vector_text",
    "save_figure",
    "save_svg_png",
    # Cross section
    "cross_section_plot",
    "cross_section_area_plot",
    # Time series
    "plot_loss_curve",
    "plot_predictions_vs_observations",
    "plot_time_series_comparison",
    "plot_metrics_summary",
    # Hydrology (model-/scale-agnostic)
    "plot_hydrograph",
    "plot_precip_hydrograph",
    "plot_event_panel",
    "plot_flow_duration_curve",
    "plot_metric_bars",
]
