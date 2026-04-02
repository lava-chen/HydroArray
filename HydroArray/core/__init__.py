"""
核心数据结构模块

提供水文数据的基础数据结构定义。
"""

from HydroArray.core.containers import (
    HydroData,
    GriddedData,
    StationData,
    as_hydrodata,
)

__all__ = [
    "HydroData",
    "GriddedData",
    "StationData",
    "as_hydrodata",
]
