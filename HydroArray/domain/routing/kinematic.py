"""
运动波汇流

使用运动波（Kinematic Wave）方法进行汇流演算。
基于 Manning 公式计算水流传播。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from HydroArray.domain.models.base import (
    GridNode,
    ModelParameters,
    RoutingModel,
)


@dataclass
class KinematicRoutingParameters(ModelParameters):
    """运动波汇流参数"""
    # Manning 粗糙系数
    n: float = 0.05

    # 坡度
    slope: float = 0.01

    # 水力宽度 (m)
    width: float = 10.0

    # 波速系数
    wave_velocity_coeff: float = 5.0 / 3.0  # = 5/3 for kinematic wave

    def validate(self) -> tuple:
        if self.n <= 0:
            return False, f"Manning n 必须为正，得到 {self.n}"
        if self.slope <= 0:
            return False, f"坡度必须为正，得到 {self.slope}"
        return True, "OK"


class KinematicRouting(RoutingModel):
    """运动波汇流模型

    基于运动波理论的汇流演算：
    Q = (1/n) * A * R^(2/3) * S^(1/2)

    其中：
    - n: Manning 粗糙系数
    - A: 过水面积
    - R: 水力半径
    - S: 坡度
    """

    def __init__(self):
        self._nodes: List[GridNode] = []
        self._params: Optional[KinematicRoutingParameters] = None
        self._n_nodes: int = 0

        # 状态
        self._flow_accumulation: Optional[np.ndarray] = None
        self._travel_time: Optional[np.ndarray] = None
        self._downstream_mask: Optional[np.ndarray] = None

        # Gamma 分布参数用于单位线
        self._gamma_shape: float = 2.0
        self._gamma_scale: float = 1.0

    def initialize(self, nodes: List[GridNode],
                   parameters: ModelParameters) -> bool:
        """初始化运动波汇流模型"""
        if not isinstance(parameters, KinematicRoutingParameters):
            raise TypeError(f"Expected KinematicRoutingParameters, got {type(parameters)}")

        # Reject invalid parameters (e.g. Manning n<=0) at init time.
        self._validate_parameters(parameters)

        self._nodes = nodes
        self._params = parameters
        self._n_nodes = len(nodes)

        # 计算汇流面积
        self._flow_accumulation = np.array([n.flow_accumulation for n in nodes])
        self._flow_accumulation = np.maximum(self._flow_accumulation, 1)

        # 构建下游掩码
        self._downstream_mask = np.full(self._n_nodes, -1, dtype=int)
        for i, node in enumerate(self._nodes):
            if node.downstream_id >= 0 and node.downstream_id < self._n_nodes:
                self._downstream_mask[i] = node.downstream_id

        # 计算传播时间（基于汇流面积）
        self._compute_travel_times()

        return True

    def _compute_travel_times(self):
        """计算各节点的传播时间"""
        # 简单近似：传播时间与汇流面积的 0.6 次方成正比
        # t = k * A^0.6 / V
        n = self._params.n
        slope = self._params.slope
        v = (1 / n) * (slope ** 0.5)  # 默认流速

        # 归一化汇流面积
        area_norm = self._flow_accumulation / self._flow_accumulation.max()

        # 传播时间（小时）
        # 假设网格大小为 1km，传播时间与面积成正比
        self._travel_time = 1.0 * area_norm / max(v, 0.1)

    def route(self, step_hours: float, inflow: np.ndarray) -> np.ndarray:
        """运动波汇流演算

        使用线性水库近似运动波。每节点按传播时间计算消退系数 k：
        - 比例 k 流向出口（出口节点直接产生 outflow；非出口节点累加
          到下游节点的入流缓存中）
        - 比例 (1-k) 残留在本节点，等待下一轮流出

        这样保证流量严格按拓扑序沿河网传递，残留在多步中排空。

        Args:
            step_hours: 时间步长（小时）
            inflow: 入流数组

        Returns:
            出流数组（出口节点非零）
        """
        inflow = np.asarray(inflow)
        if inflow.shape != (self._n_nodes,):
            inflow = np.broadcast_to(inflow.mean(), self._n_nodes)

        # Initialize the in-transit buffer on first call.
        if not hasattr(self, "_pending_inflow") or self._pending_inflow is None:
            self._pending_inflow = np.zeros(self._n_nodes)

        # Lateral inflow this step = explicit inflow + whatever was in-transit
        # from upstream nodes in previous steps.
        downstream_inflow = self._pending_inflow + inflow
        n = self._n_nodes
        outflow = np.zeros(n)
        new_pending = np.zeros(n)

        for i in range(n):
            tt = max(self._travel_time[i], 0.1)
            k = min(step_hours / tt, 0.99)  # CFL-like stability cap

            # k 比例流向出口/下游；(1-k) 残留在本节点
            q_in_i = downstream_inflow[i]
            q_out = k * q_in_i
            residual = (1.0 - k) * q_in_i

            down_idx = self._downstream_mask[i]
            if down_idx >= 0:
                # 流向下游：加到下游下一轮的入流
                new_pending[down_idx] += q_out
            else:
                # 出口：产生 outflow
                outflow[i] = q_out
                # 出口节点的残留丢弃（这是简化运动波的局限）

            # 残留留在本节点下一轮再排
            new_pending[i] += residual

        self._pending_inflow = new_pending
        return outflow

    def route_kinematic_wave(self, step_hours: float,
                            inflow: np.ndarray) -> np.ndarray:
        """完整的运动波演算（显式有限差分）"""
        inflow = np.asarray(inflow)
        n = self._params.n
        slope = self._params.slope
        width = self._params.width

        # 计算波速
        # V = (1/n) * R^(2/3) * S^(1/2)，简化 R = depth，假设均匀流
        velocity = (1 / n) * (slope ** 0.5)

        # 波速（简化）
        wave_speed = self._params.wave_velocity_coeff * velocity

        # 出流 = 入流 * (1 - 延迟因子)
        # 使用 Courant 数
        dx = 1000  # 假设网格大小 1km
        courant = wave_speed * step_hours / dx

        # 简化处理
        outflow = inflow * np.exp(-courant)

        return np.maximum(outflow, 0)

    def get_states(self) -> Dict[str, np.ndarray]:
        """获取状态"""
        return {
            'flow_accumulation': self._flow_accumulation.copy(),
            'travel_time': self._travel_time.copy() if self._travel_time is not None else None,
        }

    def set_states(self, states: Dict[str, np.ndarray]):
        """设置状态"""
        if 'flow_accumulation' in states:
            self._flow_accumulation = states['flow_accumulation'].copy()
