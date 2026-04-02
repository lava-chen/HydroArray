"""
流域蒸散发量计算

提供三层蒸发模型计算功能。
"""

import pandas as pd
import numpy as np


def three_layer_evaporation(
    Ep: float,
    WU: float,
    WL: float,
    P: float,
    WLM: float,
    C: float = 1/6,
) -> tuple[float, float, float]:
    """
    三层蒸发模型计算

    按照先上层后下层的次序，分四种情况计算蒸发量。

    Args:
        Ep: 流域蒸发能力 (mm)
        WU: 上层土壤含水量 (mm)
        WL: 下层土壤含水量 (mm)
        P: 降雨量 (mm)
        WLM: 下层土壤含水容量 (mm)
        C: 蒸发扩散系数，默认 1/6

    Returns:
        tuple[float, float, float]: (EU, EL, ED) - 上层、下层、深层蒸发量 (mm)
    """
    EU, EL, ED = 0.0, 0.0, 0.0

    if WU + P >= Ep:
        EU = Ep
        EL = 0
        ED = 0
    elif WL >= C * WLM:
        EU = WU + P
        EL = (Ep - EU) * WL / WLM
        ED = 0
    elif C * (Ep - (WU + P)) <= WL < C * WLM:
        EU = WU + P
        EL = C * (Ep - EU)
        ED = 0
    else:
        EU = WU + P
        EL = WL
        ED = C * (Ep - EU) - EL

    return EU, EL, ED
