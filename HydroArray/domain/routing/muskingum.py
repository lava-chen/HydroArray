"""
马斯京根汇流

使用马斯京根（Muskingum）方法进行河道汇流演算。
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
class MuskingumParameters(ModelParameters):
    """马斯京根汇流参数

    Args:
        K: 演算参数（相当于传播时间），小时
        x: 权重因子，0-0.5
            - x = 0: 纯调洪（最大调蓄）
            - x = 0.5: 典型洪水波
            - x > 0.5: 加速效应（非常规）
    """
    K: float = 10.0
    x: float = 0.2

    def validate(self) -> tuple:
        if self.K <= 0:
            return False, f"K 必须为正，得到 {self.K}"
        if self.x < 0 or self.x > 0.5:
            return False, f"x 必须在 [0, 0.5] 范围内，得到 {self.x}"
        return True, "OK"


class MuskingumRouting(RoutingModel):
    """马斯京根汇流模型

    马斯京根法是最常用的河道汇流方法之一。

    基本公式：
    Q_out(t) = C0 * I(t) + C1 * I(t-1) + C2 * Q_out(t-1)

    其中：
    C0 = (0.5*dt - K*x) / (K*(1-x) + 0.5*dt)
    C1 = (0.5*dt + K*x) / (K*(1-x) + 0.5*dt)
    C2 = (K*(1-x) - 0.5*dt) / (K*(1-x) + 0.5*dt)

    C0 + C1 + C2 = 1

    适用于单个河段或河网。
    """

    def __init__(self):
        self._nodes: List[GridNode] = []
        self._params: Optional[MuskingumParameters] = None
        self._n_reaches: int = 0

        # 马斯京根系数
        self._C0: float = 0.0
        self._C1: float = 0.0
        self._C2: float = 0.0

        # 状态
        self._Q_in_history: List[np.ndarray] = []  # 入流历史
        self._Q_out_history: List[np.ndarray] = []  # 出流历史

    def initialize(self, nodes: List[GridNode],
                   parameters: ModelParameters) -> bool:
        """初始化马斯京根汇流模型

        Args:
            nodes: 河段/网格节点
            parameters: 马斯京根参数
        """
        self._nodes = nodes
        self._params = parameters
        self._n_reaches = len(nodes)

        # 计算系数
        self._compute_coefficients(step_hours=1.0)

        # 初始化历史
        self._Q_in_history = [np.zeros(self._n_reaches)]
        self._Q_out_history = [np.zeros(self._n_reaches)]

        return True

    def _compute_coefficients(self, step_hours: float):
        """计算马斯京根系数"""
        K = self._params.K
        x = self._params.x
        dt = step_hours

        denom = K * (1 - x) + 0.5 * dt

        self._C0 = (0.5 * dt - K * x) / denom
        self._C1 = (0.5 * dt + K * x) / denom
        self._C2 = (K * (1 - x) - 0.5 * dt) / denom

    def route(self, step_hours: float, inflow: np.ndarray) -> np.ndarray:
        """马斯京根汇流演算

        Args:
            step_hours: 时间步长（小时）
            inflow: 入流数组 (mm 或 m³/s)

        Returns:
            出流数组
        """
        # 重新计算系数（如果 dt 变化）
        self._compute_coefficients(step_hours)

        inflow = np.asarray(inflow)
        if inflow.shape != (self._n_reaches,):
            inflow = np.broadcast_to(inflow.mean(), self._n_reaches)

        # 获取上一步的入流和出流
        Q_in_prev = self._Q_in_history[-1] if self._Q_in_history else np.zeros_like(inflow)
        Q_out_prev = self._Q_out_history[-1] if self._Q_out_history else np.zeros_like(inflow)

        # 马斯京根公式
        Q_out = self._C0 * inflow + self._C1 * Q_in_prev + self._C2 * Q_out_prev

        # 更新历史
        self._Q_in_history.append(inflow.copy())
        self._Q_out_history.append(Q_out.copy())

        # 保持历史长度合理
        if len(self._Q_in_history) > 100:
            self._Q_in_history.pop(0)
            self._Q_out_history.pop(0)

        return Q_out

    def route_segment(self, step_hours: float, I: np.ndarray) -> np.ndarray:
        """对单个河段进行汇流演算

        Args:
            step_hours: 时间步长
            I: 入流时间序列 (n_steps,)

        Returns:
            出流时间序列 (n_steps,)
        """
        n_steps = len(I)
        O = np.zeros(n_steps)

        Q_in_prev = 0.0
        Q_out_prev = 0.0

        self._compute_coefficients(step_hours)

        for t in range(n_steps):
            O[t] = self._C0 * I[t] + self._C1 * Q_in_prev + self._C2 * Q_out_prev
            Q_in_prev = I[t]
            Q_out_prev = O[t]

        return O

    def get_states(self) -> Dict[str, np.ndarray]:
        """获取状态"""
        return {
            'Q_in_history': np.array(self._Q_in_history) if self._Q_in_history else None,
            'Q_out_history': np.array(self._Q_out_history) if self._Q_out_history else None,
            'C0': np.array([self._C0]),
            'C1': np.array([self._C1]),
            'C2': np.array([self._C2]),
        }

    def set_states(self, states: Dict[str, np.ndarray]):
        """设置状态"""
        if 'Q_in_history' in states and states['Q_in_history'] is not None:
            self._Q_in_history = list(states['Q_in_history'])
        if 'Q_out_history' in states and states['Q_out_history'] is not None:
            self._Q_out_history = list(states['Q_out_history'])


class MuskingumChannel(MuskingumRouting):
    """马斯京根河道

    扩展的马斯京根模型，支持多参数（每个河段不同参数）。
    """

    def __init__(self):
        super().__init__()
        self._K_per_reach: Optional[np.ndarray] = None
        self._x_per_reach: Optional[np.ndarray] = None

    def initialize_with_reach_params(self, nodes: List[GridNode],
                                      K_values: np.ndarray,
                                      x_values: np.ndarray):
        """使用每河段参数初始化

        Args:
            nodes: 河段节点
            K_values: 每河段的K值 (n_reaches,)
            x_values: 每河段的x值 (n_reaches,)
        """
        self._nodes = nodes
        self._n_reaches = len(nodes)
        self._K_per_reach = np.asarray(K_values)
        self._x_per_reach = np.asarray(x_values)

        self._Q_in_history = [np.zeros(self._n_reaches)]
        self._Q_out_history = [np.zeros(self._n_reaches)]

    def route(self, step_hours: float, inflow: np.ndarray) -> np.ndarray:
        """使用河段特定参数进行汇流"""
        inflow = np.asarray(inflow)
        if inflow.shape != (self._n_reaches,):
            inflow = np.broadcast_to(inflow.mean(), self._n_reaches)

        Q_out = np.zeros_like(inflow)

        for i in range(self._n_reaches):
            K = self._K_per_reach[i] if self._K_per_reach is not None else self._params.K
            x = self._x_per_reach[i] if self._x_per_reach is not None else self._params.x

            dt = step_hours
            denom = K * (1 - x) + 0.5 * dt

            C0 = (0.5 * dt - K * x) / denom
            C1 = (0.5 * dt + K * x) / denom
            C2 = (K * (1 - x) - 0.5 * dt) / denom

            Q_in_prev = self._Q_in_history[-1][i] if self._Q_in_history else 0
            Q_out_prev = self._Q_out_history[-1][i] if self._Q_out_history else 0

            Q_out[i] = C0 * inflow[i] + C1 * Q_in_prev + C2 * Q_out_prev

        self._Q_in_history.append(inflow.copy())
        self._Q_out_history.append(Q_out.copy())

        if len(self._Q_in_history) > 100:
            self._Q_in_history.pop(0)
            self._Q_out_history.pop(0)

        return Q_out
