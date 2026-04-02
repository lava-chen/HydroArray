"""
线性水库汇流

使用线性水库方法进行汇流演算。
每个网格的水流按线性消退关系传递到下游。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from HydroArray.domain.models.base import (
    GridNode,
    ModelParameters,
    RoutingModel,
)


@dataclass
class LinearRoutingParameters(ModelParameters):
    """线性水库汇流参数"""
    # 各层（地表、壤中流、地下）的消退系数 (1/day)
    k_overland: float = 0.5
    k_interflow: float = 0.3
    k_baseflow: float = 0.05

    # 滞后时间（小时）
    lag_overland: float = 0.0
    lag_interflow: float = 0.0
    lag_baseflow: float = 0.0

    def validate(self) -> tuple:
        for k in [self.k_overland, self.k_interflow, self.k_baseflow]:
            if k <= 0 or k > 1:
                return False, f"消退系数必须在 (0, 1] 范围内，得到 {k}"
        return True, "OK"


class LinearRouting(RoutingModel):
    """线性水库汇流模型

    使用线性水库串联模拟水流传播：
    Q_out = K * S（线性假设）
    其中 K 是消退系数，S 是蓄量。

    汇流层次：
    - 地表径流
    - 壤中流
    - 地下径流/基流
    """

    def __init__(self):
        self._nodes: List[GridNode] = []
        self._params: Optional[LinearRoutingParameters] = None
        self._n_nodes: int = 0

        # 状态变量
        self._storage_overland: Optional[np.ndarray] = None
        self._storage_interflow: Optional[np.ndarray] = None
        self._storage_baseflow: Optional[np.ndarray] = None

        # 下游关系矩阵
        self._downstream_mask: Optional[np.ndarray] = None

    def initialize(self, nodes: List[GridNode],
                   parameters: ModelParameters) -> bool:
        """初始化汇流模型"""
        self._nodes = nodes
        self._params = parameters
        self._n_nodes = len(nodes)

        # 初始化状态
        self._storage_overland = np.zeros(self._n_nodes)
        self._storage_interflow = np.zeros(self._n_nodes)
        self._storage_baseflow = np.zeros(self._n_nodes)

        # 构建下游掩码
        self._build_downstream_mask()

        return True

    def _build_downstream_mask(self):
        """构建下游关系矩阵"""
        self._downstream_mask = np.full(self._n_nodes, -1, dtype=int)
        for i, node in enumerate(self._nodes):
            if node.downstream_id >= 0 and node.downstream_id < self._n_nodes:
                self._downstream_mask[i] = node.downstream_id

    def route(self, step_hours: float, inflow: np.ndarray) -> np.ndarray:
        """汇流演算

        Args:
            step_hours: 时间步长（小时）
            inflow: 入流数组 (mm)

        Returns:
            出流数组 (mm)
        """
        inflow = np.asarray(inflow)
        if inflow.shape != (self._n_nodes,):
            inflow = np.broadcast_to(inflow.mean(), self._n_nodes)

        dt = step_hours / 24.0  # 转换为天

        k_o = self._params.k_overland
        k_i = self._params.k_interflow
        k_b = self._params.k_baseflow

        # 计算出流（线性水库公式）
        q_overland = k_o * self._storage_overland
        q_interflow = k_i * self._storage_interflow
        q_baseflow = k_b * self._storage_baseflow

        # 更新蓄量
        self._storage_overland += inflow - q_overland * dt
        self._storage_interflow += inflow - q_interflow * dt
        self._storage_baseflow += inflow - q_baseflow * dt

        # 确保非负
        self._storage_overland = np.maximum(self._storage_overland, 0)
        self._storage_interflow = np.maximum(self._storage_interflow, 0)
        self._storage_baseflow = np.maximum(self._storage_baseflow, 0)

        # 向下游传递（简单延迟传递）
        outflow = np.zeros(self._n_nodes)
        for i in range(self._n_nodes):
            down_idx = self._downstream_mask[i]
            if down_idx >= 0:
                # 地表和壤中流传递到下游
                self._storage_overland[down_idx] += q_overland[i] * dt
                self._storage_interflow[down_idx] += q_interflow[i] * dt
            else:
                # 出口节点
                outflow[i] = q_overland[i] + q_interflow[i] + q_baseflow[i]

        return outflow

    def route_total(self, step_hours: float, total_inflow: np.ndarray) -> np.ndarray:
        """直接对总入流进行汇流演算（简化模式）"""
        inflow = np.asarray(total_inflow)
        if inflow.shape != (self._n_nodes,):
            inflow = np.broadcast_to(inflow.mean(), self._n_nodes)

        dt = step_hours / 24.0
        k = (self._params.k_overland + self._params.k_interflow +
             self._params.k_baseflow) / 3.0

        q_out = k * self._storage_overland
        self._storage_overland += inflow - q_out * dt
        self._storage_overland = np.maximum(self._storage_overland, 0)

        # 下游传递
        outflow = np.zeros(self._n_nodes)
        for i in range(self._n_nodes):
            down_idx = self._downstream_mask[i]
            if down_idx >= 0:
                self._storage_overland[down_idx] += q_out[i] * dt
            else:
                outflow[i] = q_out[i]

        return outflow

    def get_states(self) -> Dict[str, np.ndarray]:
        """获取状态"""
        return {
            'storage_overland': self._storage_overland.copy(),
            'storage_interflow': self._storage_interflow.copy(),
            'storage_baseflow': self._storage_baseflow.copy(),
        }

    def set_states(self, states: Dict[str, np.ndarray]):
        """设置状态"""
        if 'storage_overland' in states:
            self._storage_overland = states['storage_overland'].copy()
        if 'storage_interflow' in states:
            self._storage_interflow = states['storage_interflow'].copy()
        if 'storage_baseflow' in states:
            self._storage_baseflow = states['storage_baseflow'].copy()
