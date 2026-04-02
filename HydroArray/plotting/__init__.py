"""
可视化模块
"""

from HydroArray.plotting.styles import use_hydro_style, get_colors, list_styles

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

__all__ = [
    # Styles
    "use_hydro_style",
    "get_colors",
    "list_styles",
    # Cross section
    "cross_section_plot",
    "cross_section_area_plot",
    # Time series
    "plot_loss_curve",
    "plot_predictions_vs_observations",
    "plot_time_series_comparison",
    "plot_metrics_summary",
]
