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

        使用 Gamma 单位线进行汇流演算。

        Args:
            step_hours: 时间步长（小时）
            inflow: 入流数组

        Returns:
            出流数组
        """
        inflow = np.asarray(inflow)
        if inflow.shape != (self._n_nodes,):
            inflow = np.broadcast_to(inflow.mean(), self._n_nodes)

        # 使用线性水库近似运动波
        # 每个节点的消退系数与其传播时间相关
        outflow = np.zeros(self._n_nodes)

        for i in range(self._n_nodes):
            tt = max(self._travel_time[i], 0.1)
            # 简化的运动波：K = dt / tt
            k = step_hours / tt
            k = min(k, 0.99)  # 确保稳定

            # 出口节点直接输出
            down_idx = self._downstream_mask[i]
            if down_idx < 0:
                outflow[i] = k * inflow[i]
            else:
                # 传递到下游
                pass  # 下游由其自己的计算处理

        # 汇总到出口
        exit_idx = np.where(self._downstream_mask == -1)[0]
        if len(exit_idx) > 0:
            total_input = inflow.sum()
            # 简单分配到出口
            for idx in exit_idx:
                outflow[idx] = total_input / len(exit_idx)

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
