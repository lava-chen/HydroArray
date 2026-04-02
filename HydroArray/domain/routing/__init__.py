"""
汇流模块

提供各种汇流演算方法：线性水库、马斯京根、运动波等。
"""

from HydroArray.domain.routing.linear import LinearRouting, LinearRoutingParameters
from HydroArray.domain.routing.kinematic import KinematicRouting, KinematicRoutingParameters
from HydroArray.domain.routing.muskingum import MuskingumRouting, MuskingumParameters

__all__ = [
    "LinearRouting",
    "LinearRoutingParameters",
    "KinematicRouting",
    "KinematicRoutingParameters",
    "MuskingumRouting",
    "MuskingumParameters",
]
