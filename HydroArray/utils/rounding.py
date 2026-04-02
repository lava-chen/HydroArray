"""
修约函数

遵循水文测验规范进行数值修约。
"""

import numpy as np


def round_to_n_sig_figs(x: float, n: int) -> float:
    """将数字修约到 n 位有效数字"""
    if x == 0:
        return 0
    return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))


def round_area(area: float) -> float:
    """断面面积修约：取三位有效数字，但小数不过二位"""
    if area == 0:
        return 0
    
    # 先取三位有效数字
    rounded = round_to_n_sig_figs(area, 3)
    
    # 确定小数位数：保证不超过2位，同时尽量保留有效数字
    # 根据数值大小决定小数位数
    if area >= 100:
        # 大于等于100时，取整数（如 18100）
        return round(rounded)
    elif area >= 10:
        # 10-100之间，保留1位小数（如 18.6）
        return round(rounded, 1)
    else:
        # 小于10，保留2位小数（如 5.71）
        return round(rounded, 2)


def round_distance(dist: float) -> float:
    """起点距修约规则：
    - 大于或等于 100m 时，记至 1m（整数）
    - 小于 100m 时，小数不过一位
    """
    if dist >= 100:
        return round(dist)
    else:
        return round(dist, 1)


def round_width(width: float) -> float:
    """水面宽修约规则：
    - 优先取三位有效数字
    - 小于 5m 时，小数不过二位
    - 大于或等于 5m 且小于 10m 时，小数不过一位
    - 大于或等于 10m 时，按三位有效数字修约
    """
    if width == 0:
        return 0
    rounded = round_to_n_sig_figs(width, 3)
    if width < 5:
        return round(rounded, 2)
    elif width < 10:
        return round(rounded, 1)
    else:
        return rounded
