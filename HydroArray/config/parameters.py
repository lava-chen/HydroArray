"""
参数管理

提供水文模型参数的注册和管理功能。
"""

from dataclasses import dataclass, field
from typing import Dict, Type, Any, Optional, Tuple
import inspect

from HydroArray.domain.models.base import ModelParameters


# 全局参数注册表
_PARAM_REGISTRY: Dict[str, Type[ModelParameters]] = {}


def register_parameters(name: str = None):
    """参数类注册装饰器

    Args:
        name: 注册名称，默认为类名小写

    Example:
        @register_parameters()
        class XinAnjiangParams(ModelParameters):
            WM: float = 100.0
    """
    def decorator(cls: Type[ModelParameters]) -> Type[ModelParameters]:
        nonlocal name
        if name is None:
            name = cls.__name__.lower()

        # 验证是 ModelParameters 的子类
        if not issubclass(cls, ModelParameters):
            raise ValueError(f"{cls} 必须继承 ModelParameters")

        _PARAM_REGISTRY[name] = cls
        return cls

    return decorator


def get_parameters_class(name: str) -> Optional[Type[ModelParameters]]:
    """获取已注册的参数类

    Args:
        name: 注册名称

    Returns:
        参数类或 None
    """
    return _PARAM_REGISTRY.get(name)


def list_registered_parameters() -> list:
    """列出所有注册的参数类"""
    return list(_PARAM_REGISTRY.keys())


class ParametersManager:
    """参数管理器

    统一管理模型参数的创建、验证和序列化。
    """

    def __init__(self):
        self._param_instances: Dict[str, ModelParameters] = {}

    def register(self, name: str, param_class: Type[ModelParameters]):
        """注册参数类"""
        _PARAM_REGISTRY[name] = param_class

    def create(self, name: str, **kwargs) -> ModelParameters:
        """创建参数实例

        Args:
            name: 参数类名称
            **kwargs: 参数值

        Returns:
            参数实例
        """
        param_class = get_parameters_class(name)
        if param_class is None:
            raise ValueError(f"未找到参数类: {name}")

        return param_class(**kwargs)

    def from_dict(self, name: str, params: Dict[str, float]) -> ModelParameters:
        """从字典创建参数

        Args:
            name: 参数类名称
            params: 参数字典
        """
        param_class = get_parameters_class(name)
        if param_class is None:
            raise ValueError(f"未找到参数类: {name}")

        return param_class.from_dict(params)

    def validate(self, params: ModelParameters) -> Tuple[bool, str]:
        """验证参数

        Args:
            params: 参数实例

        Returns:
            (是否有效, 消息)
        """
        return params.validate()

    def to_dict(self, params: ModelParameters) -> Dict[str, float]:
        """参数转字典"""
        return params.to_dict()

    def save(self, params: ModelParameters, path: str):
        """保存参数到文件"""
        import json
        data = self.to_dict(params)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, name: str, path: str) -> ModelParameters:
        """从文件加载参数"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return self.from_dict(name, data)


@dataclass
class ParametersBundle:
    """参数包

    用于批量管理多个模型的参数。
    """
    name: str = ""
    water_balance_params: Optional[ModelParameters] = None
    routing_params: Optional[ModelParameters] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {'name': self.name}
        if self.water_balance_params:
            result['water_balance'] = self.water_balance_params.to_dict()
        if self.routing_params:
            result['routing'] = self.routing_params.to_dict()
        result['extra'] = self.extra_params
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParametersBundle":
        """从字典创建"""
        bundle = cls(name=data.get('name', ''))

        if 'water_balance' in data:
            # 需要调用者提供参数类名来实例化
            pass

        bundle.extra_params = data.get('extra', {})
        return bundle
