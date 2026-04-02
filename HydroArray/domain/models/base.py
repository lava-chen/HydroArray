"""
Hydrological Model Base Classes

This module defines the abstract base classes for all hydrological models:
- WaterBalanceModel: For water balance / runoff generation
- RoutingModel: For flow routing in river networks
- HydrologyModel: Combined water balance + routing

Example:
    >>> from HydroArray.domain.models import WaterBalanceModel, GridNode
    >>> class MyModel(WaterBalanceModel):
    ...     def initialize(self, nodes, parameters) -> bool:
    ...         ...
    ...     def water_balance(self, step_hours, precip, pet) -> tuple:
    ...         ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from HydroArray.core.containers import GriddedData, StationData


# Global parameter registry
_PARAM_REGISTRY: Dict[str, type] = {}


@dataclass
class ModelParameters:
    """Base class for hydrological model parameters.

    All model parameter classes should inherit from this class.
    Parameters can be serialized to/from dictionaries and validated.

    Example:
        >>> @register_parameters("my_model")
        ... @dataclass
        ... class MyModelParameters(ModelParameters):
        ...     WM: float = 100.0
        ...     K: float = 0.5
    """

    @classmethod
    def from_dict(cls, params: Dict[str, float]) -> "ModelParameters":
        """Create parameter instance from dictionary.

        Args:
            params: Dictionary of parameter names and values.

        Returns:
            New parameter instance.
        """
        return cls(**params)

    def to_dict(self) -> Dict[str, float]:
        """Convert parameters to dictionary.

        Returns:
            Dictionary mapping parameter names to values.
        """
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and not callable(value):
                result[key] = value
        return result

    def validate(self) -> Tuple[bool, str]:
        """Validate parameter values.

        Returns:
            Tuple of (is_valid, message).
        """
        return True, "OK"


@dataclass
class GridNode:
    """Grid node for spatial discretization in distributed hydrological models.

    Attributes:
        node_id: Unique identifier for the node.
        x: X coordinate (e.g., longitude or cartesian).
        y: Y coordinate (e.g., latitude or cartesian).
        area: Catchment area (km²).
        elevation: Elevation (m).
        downstream_id: ID of the downstream node (-1 for outlet).
        flow_direction: D8 flow direction code.
        flow_accumulation: Number of upstream cells contributing to this node.
        stream_order: Strahler stream order.
        lat: Latitude (defaults to y).
        lon: Longitude (defaults to x).

    Example:
        >>> node = GridNode(node_id=0, x=100.5, y=35.2, area=150.0)
        >>> node.downstream_id = 1  # Set downstream node
    """
    node_id: int = 0
    x: float = 0.0
    y: float = 0.0
    area: float = 1.0  # km²
    elevation: float = 0.0  # m
    downstream_id: int = -1  # downstream node ID, -1 for outlet
    flow_direction: int = -1  # D8 flow direction code
    flow_accumulation: int = 0  # contributing cells
    stream_order: int = 0  # Strahler order
    lat: Optional[float] = None
    lon: Optional[float] = None

    def __post_init__(self):
        if self.lat is None:
            self.lat = self.y
        if self.lon is None:
            self.lon = self.x


class WaterBalanceModel(ABC):
    """Abstract base class for water balance (runoff generation) models.

    All runoff production models (e.g., XinAnjiang, SAC-SMA, HyMOD)
    must implement this interface.

    The model computes:
    1. Evapotranspiration losses
    2. Runoff generation (saturation excess, infiltration excess, etc.)
    3. Source separation (surface, interflow, groundwater)

    Example:
        >>> class MyModel(WaterBalanceModel):
        ...     def initialize(self, nodes, parameters) -> bool:
        ...         self._nodes = nodes
        ...         self._params = parameters
        ...         return True
        ...     def water_balance(self, step_hours, precip, pet):
        ...         # Compute runoff components
        ...         return surface, interflow, baseflow, extra_states

    Args:
        ABC: Abstract base class from abc module.
    """

    @abstractmethod
    def initialize(self, nodes: List[GridNode], parameters: ModelParameters) -> bool:
        """Initialize the model with nodes and parameters.

        Args:
            nodes: List of GridNode objects defining the spatial discretization.
            parameters: ModelParameters instance with model-specific values.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def water_balance(self, step_hours: float, precip: np.ndarray,
                      pet: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Compute water balance and runoff components.

        Args:
            step_hours: Time step duration in hours.
            precip: Precipitation array (mm), shape (n_nodes,).
            pet: Potential evapotranspiration array (mm), shape (n_nodes,).

        Returns:
            Tuple of (surface_runoff, interflow, baseflow, extra_states):
            - surface_runoff: Surface runoff (mm)
            - interflow: Interflow/壤中流 (mm)
            - baseflow: Groundwater/baseflow/地下径流 (mm)
            - extra_states: Additional state variables (e.g., soil moisture)
        """
        pass

    @abstractmethod
    def get_states(self) -> Dict[str, np.ndarray]:
        """Get current model state variables.

        Returns:
            Dictionary mapping state variable names to arrays.
        """
        pass

    @abstractmethod
    def set_states(self, states: Dict[str, np.ndarray]):
        """Set model state variables.

        Args:
            states: Dictionary mapping state variable names to arrays.
        """
        pass

    def save_states(self, path: str):
        """Save model states to file.

        Args:
            path: Output file path (.npz format).
        """
        states = self.get_states()
        np.savez(path, **states)

    def load_states(self, path: str):
        """Load model states from file.

        Args:
            path: Input file path (.npz format).
        """
        data = np.load(path, allow_pickle=True)
        self.set_states(dict(data))


class RoutingModel(ABC):
    """Abstract base class for flow routing models.

    Routing models transform runoff from different locations
    to the watershed outlet using channel/hillslope routing.

    Implementations should provide:
    - Linear reservoir routing
    - Kinematic wave routing
    - Muskingum channel routing
    """

    @abstractmethod
    def initialize(self, nodes: List[GridNode], parameters: ModelParameters) -> bool:
        """初始化汇流模型

        Args:
            nodes: 网格节点列表
            parameters: 汇流参数

        Returns:
            初始化是否成功
        """
        pass

    @abstractmethod
    def route(self, step_hours: float, inflow: np.ndarray) -> np.ndarray:
        """汇流演算

        Args:
            step_hours: 时间步长（小时）
            inflow: 入流量数组 (mm 或 m³/s，取决于模型)

        Returns:
            出流量数组
        """
        pass

    @abstractmethod
    def get_states(self) -> Dict[str, np.ndarray]:
        """获取当前模型状态"""
        pass

    @abstractmethod
    def set_states(self, states: Dict[str, np.ndarray]):
        """设置模型状态"""
        pass


class HydrologyModel(ABC):
    """完整水文模型

    组合水量平衡模型和汇流模型，提供完整的流域模拟功能。
    """

    def __init__(self, water_balance: WaterBalanceModel,
                 routing: Optional[RoutingModel] = None):
        """初始化完整水文模型

        Args:
            water_balance: 水量平衡模型
            routing: 汇流模型（可选）
        """
        self.wb_model = water_balance
        self.routing = routing
        self._is_initialized = False

    @abstractmethod
    def run(self, inputs: GriddedData) -> GriddedData:
        """运行完整模型

        Args:
            inputs: 输入数据（降水、蒸发等）

        Returns:
            模拟结果
        """
        pass

    @abstractmethod
    def run_step(self, step_hours: float, precip: np.ndarray,
                 pet: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """单步计算

        Args:
            step_hours: 时间步长
            precip: 降水量
            pet: 潜在蒸发量

        Returns:
            (discharge, states)
        """
        pass

    def calibrate(self, inputs: GriddedData, observations: StationData):
        """模型率定

        Args:
            inputs: 输入数据
            observations: 观测数据
        """
        raise NotImplementedError("率定功能待实现")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "water_balance": self.wb_model.__class__.__name__,
            "routing": self.routing.__class__.__name__ if self.routing else None,
            "initialized": self._is_initialized,
        }


class WaterBalanceRouted(HydrologyModel):
    """水量平衡 + 汇流组合模型

    通用组合模型，先计算产流，再进行汇流演算。
    """

    def __init__(self, water_balance: WaterBalanceModel,
                 routing: Optional[RoutingModel] = None):
        super().__init__(water_balance, routing)
        self._nodes: List[GridNode] = []

    def run(self, inputs: GriddedData) -> GriddedData:
        """运行完整模拟"""
        # 从 inputs 提取降水和蒸发数据
        precip = inputs.data['precipitation'] if isinstance(inputs.data, dict) else inputs.data
        pet = inputs.data.get('pet', np.zeros_like(precip)) if isinstance(inputs.data, dict) else np.zeros_like(precip)

        n_steps = precip.shape[0] if precip.ndim > 0 else 1
        time_coords = inputs.coords.get('time', np.arange(n_steps))

        q_sim = []
        states_history = []

        for i in range(n_steps):
            p = precip[i] if precip.ndim > 0 else precip
            e = pet[i] if pet.ndim > 0 else pet

            q, states = self.run_step(1.0, p, e)
            q_sim.append(q)
            states_history.append(states)

        # 组装结果
        from HydroArray.core.containers import StationData
        result_data = np.array(q_sim) if q_sim else np.array([])
        return StationData(
            data=result_data,
            coords={'time': time_coords},
            dims=['time'],
            name='Q_sim'
        )

    def run_step(self, step_hours: float, precip: np.ndarray,
                 pet: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """单步计算"""
        # 1. 水量平衡计算
        surface, interflow, baseflow, extra = self.wb_model.water_balance(
            step_hours, precip, pet
        )

        total_runoff = surface + interflow + baseflow

        # 2. 汇流演算
        if self.routing is not None:
            q_out = self.routing.route(step_hours, total_runoff)
        else:
            q_out = total_runoff

        return q_out, {
            'surface': surface,
            'interflow': interflow,
            'baseflow': baseflow,
            **extra
        }
