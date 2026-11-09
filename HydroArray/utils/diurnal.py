"""
Diurnal (within-day) distribution utilities.

把日值气象强迫量(蒸发、参考蒸散)降尺度到小时或更细尺度,
叠加三角函数形式的日变化,使日内分配更接近真实物理过程。
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


__all__ = [
    "diurnal_factor",
    "hourly_pet_from_daily",
]


def diurnal_factor(
    hour_of_day: Union[np.ndarray, pd.Series, float],
    amp: float = 0.3,
    peak_hour: float = 13.0,
) -> np.ndarray:
    """日变化系数: f(h) = 1 + amp · cos(2π(h - peak_hour) / 24).

    数学性质
    --------
    - 24h 累加恒为 24(余弦函数在一个完整周期上的积分为 0),即
      `sum(diurnal_factor(h for h in 0..23)) == 24`. 因此把日值
      乘以 f(h)/24 降尺度到小时,日总量严格守恒,不会改变径流深。
    - ``amp=0`` 时 f(h) ≡ 1,退化为均匀分布,等价于"日值 / 24"。
    - 当 ``amp > 0`` 时, ``f(peak_hour) = 1 + amp``(峰值时刻);
      ``f(peak_hour + 12) = 1 - amp`(谷值时刻,通常凌晨)。

    Args:
        hour_of_day: 小时数(0~24),可以是标量、``np.ndarray`` 或
            ``pd.Series``(Series 会原样保留 index)。
        amp: 振幅,``[0, 0.6]``。``0.3~0.5`` 接近中纬度陆地典型日变化;
            > 0.6 时夜间出现负的蒸发需求,物理不合理。
        peak_hour: 峰值时刻(小时,0~24)。中纬度典型 12~14 点(地方时)。

    Returns:
        与 ``hour_of_day`` 等长的系数数组。

    Example:
        >>> h = np.arange(24)
        >>> f = diurnal_factor(h, amp=0.3, peak_hour=13)
        >>> f.sum()
        24.0
    """
    h = np.asarray(hour_of_day, dtype=float)
    f = 1.0 + float(amp) * np.cos(2.0 * np.pi * (h - float(peak_hour)) / 24.0)
    if isinstance(hour_of_day, pd.Series):
        return pd.Series(f, index=hour_of_day.index, name="diurnal_factor")
    return f


def hourly_pet_from_daily(
    daily_series: pd.Series,
    hourly_index: pd.DatetimeIndex,
    amp: float = 0.3,
    peak_hour: float = 13.0,
) -> pd.Series:
    """把日值 ET 序列降尺度到小时尺度,叠加三角函数日变化。

    设 ``D[d]`` 是日值(mm/d),``H[t]`` 是小时值(mm/h),则::

        H[t] = (D[normalize(t)] / 24) * diurnal_factor(hour_of_day[t])

    由 ``diurnal_factor`` 的守恒性,任意 24 小时窗口内::

        sum(H[t] for t in [d_00:00, d_23:00])  ==  D[d]

    Args:
        daily_series: ``pd.Series``,索引为日期(``DatetimeIndex``,
            任意粒度,会自动 ``normalize`` 到 00:00),值为日值 mm/day。
        hourly_index: ``pd.DatetimeIndex``,小时级时间戳,长度通常为
            ``24 * days``。
        amp: 三角函数振幅 ``[0, 0.6]``,详见 :func:`diurnal_factor`。
        peak_hour: 三角函数峰值时刻(小时,0~24),详见 :func:`diurnal_factor`。

    Returns:
        ``pd.Series``,索引为 ``hourly_index``,单位 mm/hour。

    Raises:
        ValueError: 当 ``daily_series`` 在 ``hourly_index`` 覆盖的日期里
            存在缺失时(``reindex`` 之后出现 NaN)。

    Example:
        >>> import pandas as pd
        >>> daily = pd.Series([3.6], index=pd.to_datetime(["2021-07-01"]))
        >>> hours = pd.date_range("2021-07-01", periods=24, freq="H")
        >>> pet = hourly_pet_from_daily(daily, hours, amp=0.3, peak_hour=13)
        >>> pet.sum()  # ≈ 3.6 (日总量守恒)
    """
    if not isinstance(daily_series, pd.Series):
        raise TypeError(f"daily_series must be pd.Series, got {type(daily_series)}")
    if not isinstance(hourly_index, pd.DatetimeIndex):
        raise TypeError(f"hourly_index must be pd.DatetimeIndex, got {type(hourly_index)}")

    # 1) 日值基准: 均匀分布 D / 24
    base = daily_series.reindex(hourly_index.normalize()).values / 24.0

    if np.any(np.isnan(base)):
        missing_dates = pd.DatetimeIndex(
            hourly_index.normalize().unique()[np.isnan(base)[:24]][0:1]
        )
        raise ValueError(
            f"daily_series 在 hourly_index 覆盖的日期里存在缺失。"
            f"示例缺失日期: {missing_dates[0].date()}。"
            "请先在 daily_series 里 forward/backward fill,或传入完整的日值序列。"
        )

    # 2) 三角函数日变化系数
    hours = hourly_index.hour + hourly_index.minute / 60.0 + hourly_index.second / 3600.0
    f = diurnal_factor(hours, amp=amp, peak_hour=peak_hour)

    # 3) 乘回去
    pet_h = base * np.asarray(f)
    return pd.Series(pet_h, index=hourly_index, name="pet_h")
