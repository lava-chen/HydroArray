"""
配置管理模块

提供水文模型的参数配置和任务配置管理。
"""

from HydroArray.config.parameters import (
    ParametersManager,
    register_parameters,
    get_parameters_class,
)
from HydroArray.config.task import (
    TaskConfig,
    ModelConfig,
    ModelType,
    RoutingType,
)

__all__ = [
    "ParametersManager",
    "register_parameters",
    "get_parameters_class",
    "TaskConfig",
    "ModelConfig",
    "ModelType",
    "RoutingType",
]
