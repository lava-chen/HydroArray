"""
参数管理

提供水文模型参数的注册和管理功能。
"""

from dataclasses import dataclass, field
from typing import Dict, Type, Any, Optional, Tuple, Iterable, Mapping, Union
import inspect

from HydroArray.domain.models.base import ModelParameters
from HydroArray.domain.calibration import ParameterBounds


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


# ──────────── ref 表 → ParameterBounds 列表 ────────────

_REF_SCALAR_FACTOR = 0.5  # 单值时的区间膨胀系数: [v*(1-F), v*(1+F)]


def _parse_ref_value(raw: Union[str, float, int, None]) -> Optional[Tuple[float, float, float]]:
    """解析一行 ref_value, 返回 (lo, hi, mid) 或 None(无法解析,如 '0.7-KG')。

    支持:
        - 数字 (1, 0.3, 0.01): 单值,推 [v*(1-F), v*(1+F)]
        - 字符串单值 ("0.3"): 同上
        - 字符串区间 ("0.8~1.2" / "0.8~ 1.2"): 直接 (lo, hi)
        - None / 空字符串: 返回 None
        - 含字母变量 ("0.7-KG"): 返回 None(调用方应用 overrides)
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # 区间: a~b
    if "~" in s:
        try:
            lo_s, hi_s = s.split("~", 1)
            lo, hi = float(lo_s.strip()), float(hi_s.strip())
        except ValueError:
            return None
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi, 0.5 * (lo + hi)

    # 单值:必须纯数字
    try:
        v = float(s)
    except ValueError:
        return None
    half = abs(v) * _REF_SCALAR_FACTOR
    # v=0 时退化为 [-1, 1] 兜底
    if half == 0.0:
        half = 1.0
    return v - half, v + half, v


def bounds_from_ref_table(
    ref_table: Mapping[str, Any],
    overrides: Optional[Mapping[str, Tuple[float, float]]] = None,
    *,
    name_alias: Optional[Mapping[str, str]] = None,
) -> list:
    """从 ref 表(通常是 params_ref.yaml 展平的 dict)生成 ParameterBounds 列表。

    Args:
        ref_table: {yaml_name: ref_value, ...},如 {"K": "0.8~1.2", "B": 0.3, ...}
        overrides: yaml 里没有/无法解析的项,手填 {param_name: (lo, hi)};
                  param_name 是 XinAnjiangParameters 字段名(经 name_alias 映射后)
        name_alias: yaml 名 → 模型字段名的映射,默认 {"IM": "IMP"}

    Returns:
        list[ParameterBounds],顺序 = ref_table 出现顺序 + overrides 出现顺序

    Example:
        >>> ref = {"K": "0.8~1.2", "B": 0.3, "WM": 120, "IM": 0.01}
        >>> bounds_from_ref_table(
        ...     ref,
        ...     overrides={"KE": (1.0, 72.0)},
        ... )
        [ParameterBounds('K', 0.8, 1.2, 1.0),
         ParameterBounds('B', 0.15, 0.45, 0.3),
         ParameterBounds('WM', 60.0, 180.0, 120.0),
         ParameterBounds('IMP', 0.005, 0.015, 0.01),
         ParameterBounds('KE', 1.0, 72.0, 36.5)]
    """
    if name_alias is None:
        name_alias = {"IM": "IMP"}

    bounds: list = []
    seen: set = set()

    # 1) 按 ref_table 顺序
    for yaml_name, raw in ref_table.items():
        field_name = name_alias.get(yaml_name, yaml_name)
        parsed = _parse_ref_value(raw)
        if parsed is None:
            # 跳过无法解析的(如 "0.7-KG")
            continue
        lo, hi, mid = parsed
        bounds.append(ParameterBounds(field_name, lo, hi, default_value=mid))
        seen.add(field_name)

    # 2) 追加 overrides(yaml 里没有的项)
    if overrides:
        for field_name, (lo, hi) in overrides.items():
            if field_name in seen:
                continue  # 已在 ref 中,避免重复
            mid = 0.5 * (lo + hi)
            bounds.append(ParameterBounds(field_name, lo, hi, default_value=mid))
            seen.add(field_name)

    return bounds
