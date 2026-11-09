"""
参数归一化:在 [0,1] 空间里做优化,真实参数空间里做评估。

为什么: WM~120 和 K~1 的量级差 120 倍, L-BFGS-B 在原始空间会只动 WM 不动 K。
归一化后,所有参数对梯度的贡献量级一致。
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np

from HydroArray.domain.calibration import ParameterBounds


class Normalizer:
    """参数 ↔ [0,1] 区间互转。

    Example::

        norm = Normalizer([ParameterBounds("K", 0.5, 1.2),
                           ParameterBounds("WM", 80, 200)])
        u = norm.to_unit(np.array([0.8, 120.0]))
        # u == [0.5, 0.333...]
        x = norm.to_physical(u)
        # x == [0.8, 120.0]
    """

    def __init__(self, bounds: List[ParameterBounds]):
        if not bounds:
            raise ValueError("Normalizer requires at least 1 bound")
        self.names: List[str] = [pb.name for pb in bounds]
        self.lo: np.ndarray = np.array(
            [pb.min_value for pb in bounds], dtype=float,
        )
        self.hi: np.ndarray = np.array(
            [pb.max_value for pb in bounds], dtype=float,
        )
        span = self.hi - self.lo
        # lo==hi 时 span=0 会除零; 用 1 占位后 to_physical/u 仍工作
        self.span: np.ndarray = np.where(span > 0, span, 1.0)
        self.unit_bounds: List[ParameterBounds] = [
            ParameterBounds(n, 0.0, 1.0, default_value=0.5)
            for n in self.names
        ]

    @property
    def n_params(self) -> int:
        return len(self.names)

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x - self.lo) / self.span

    def to_physical(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        return u * self.span + self.lo

    def physical_to_dict(self, x: np.ndarray) -> dict:
        return {n: float(v) for n, v in zip(self.names, x)}

    def unit_center(self) -> np.ndarray:
        return np.full(self.n_params, 0.5)

    def to_unit_dict(self, x: np.ndarray) -> dict:
        u = self.to_unit(x)
        return {n: float(v) for n, v in zip(self.names, u)}

    def from_unit_dict(self, u_dict: dict) -> np.ndarray:
        return np.array([u_dict[n] for n in self.names], dtype=float)


def lhs_starts(
    n_starts: int, dim: int, rng: np.random.Generator
) -> np.ndarray:
    """拉丁超立方起点(Latin Hypercube Sampling)。

    每个维度切 n_starts 段,每段抽一个,再 shuffle 列。
    比纯 random 起步覆盖更均匀。
    """
    if n_starts <= 1:
        return np.full((max(1, n_starts), dim), 0.5)
    edges = np.linspace(0.0, 1.0, n_starts + 1)
    starts = np.empty((n_starts, dim))
    for d in range(dim):
        u = rng.uniform(0.0, 1.0, size=n_starts)
        starts[:, d] = edges[:-1] + u * (edges[1:] - edges[:-1])
    for i in range(n_starts):
        rng.shuffle(starts[i])
    return starts
