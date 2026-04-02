"""
断面图

提供大断面和水道断面的可视化功能。
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

from HydroArray.plotting.styles import use_hydro_style

# 应用全局主题
use_hydro_style(dpi=100)


def cross_section_area_plot(
        data_df: pd.DataFrame,
        given_elevation_list: list,
        lowest_elevation: float,
        title: str = "大断面计算结果"
    ) -> None:
    """
    绘制大断面计算结果的折线图

    Args:
        data_df: 包含起点距（默认左岸）和各测点河底高程的DataFrame
        given_elevation_list: 给定的排序河底高程列表
        lowest_elevation: 历年最低水位
        title: 图表标题
    """
    plt.figure()
    plt.plot(given_elevation_list, data_df["area(m^2)"], marker='o', linestyle='-', color='b')
    plt.xlabel("河底高程 (m)")
    plt.ylabel("面积 (m^2)")
    plt.title(title)
    plt.grid(True)
    plt.show()


def cross_section_plot(
        data_df: pd.DataFrame,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        language: str = "zh"
    ) -> None:
    """
    绘制大断面截面图

    Args:
        data_df: 包含起点距（默认左岸）和各测点河底高程的DataFrame
        given_elevation_list: 给定的排序河底高程列表
        lowest_elevation: 历年最低水位
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        language: 语言选择
    """

    if title is None:
        if language == "zh":
            title = "大断面截面图"
        else:
            title = "Cross Section Plot"
    if xlabel is None:
        if language == "zh":
            xlabel = "起点距 (m)"
        else:
            xlabel = "Distance (m)"
    if ylabel is None:
        if language == "zh":
            ylabel = "河底水位 (m)"
        else:
            ylabel = "Elevation (m)"

    try:
        if "elevation" in data_df.columns:
            elevations = data_df["elevation"].values
        if "distance" in data_df.columns:
            distance = data_df["distance"].values
    except KeyError:
        elevations = data_df["width(m)"].values
        if "distance" in data_df.columns:
            distance = data_df["distance"].values
        else:
            distance = np.arange(len(elevations))

    plt.figure()
    plt.plot(np.asarray(distance), np.asarray(elevations), marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()


def cross_section_quick_plot(
        x_data: list | np.ndarray,
        z_data: list | np.ndarray,
        results: list,
        sample_levels: list | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
        language: str = "zh"
    ) -> plt.Figure:
    """
    绘制大断面图及Z~A曲线（合并显示）

    Args:
        x_data: 起点距列表
        z_data: 高程列表
        results: 计算结果列表，每个元素为 (水位, 面积) 元组
        sample_levels: 示例水位列表
        save_path: 图片保存路径
        show: 是否显示图表
        language: 语言选择 ("zh" 或 "en")

    Returns:
        plt.Figure: matplotlib Figure 对象
    """
    if sample_levels is None:
        sample_levels = []

    labels = {
        "zh": {
            "z": "Z (m)",
            "d": "D (m)",
            "a": "A (m²)",
        },
        "en": {
            "z": "Z (m)",
            "d": "D (m)",
            "a": "A (m²)",
        }
    }

    lang = labels.get(language, labels["zh"])

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左图：断面图（带描点）
    ax1.plot(x_data, z_data, 'k-', linewidth=1, marker='o', markersize=3)

    ax1.set_xlabel(lang["d"])
    ax1.set_ylabel(lang["z"])
    ax1.set_xlim(0, max(x_data) * 1.1)
    ax1.set_ylim(0, max(z_data) * 1.1)
    ax1.grid(True, alpha=0.3)

    # 右图：Z~A曲线（共享Y轴，不描点）
    ax2 = ax1.twiny()
    elevations = [r[0] for r in results]
    areas = [r[1] for r in results]
    ax2.plot(areas, elevations, 'k-', linewidth=1)

    # 设置顶部X轴
    ax2.set_xlim(0, max(areas) * 1.1)

    # 将顶部X轴移到下方
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 40))
    ax2.set_xlabel(lang["a"])

    # 调整布局
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig
