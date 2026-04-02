"""
水位数据处理

提供水位数据的处理功能，包括瞬时水位与日平均水位的转换、水位频率分析等。
"""

import pandas as pd
import numpy as np
from typing import Literal


ARITHMETIC_DAILY_VARIATION_THRESHOLD = 0.12


def to_daily(
    instant_water_level_data: pd.DataFrame,
    daily_water_level_data: pd.DataFrame | None = None,
    station_id: str = "default",
    unit: str = "m",
) -> pd.DataFrame:
    """
    将瞬时水位数据转换为日平均水位数据。

    根据数据特征自动判断使用**算术平均法**或**面积包围法**。

    Args:
        instant_water_level_data: 瞬时水位数据，需包含 ``time`` 和 ``water_level`` 列。
        daily_water_level_data: 可选，已有的日平均水位数据，用于合并或验证。
        station_id: 站点 ID。
        unit: 水位单位，默认为 "m"。

    Returns:
        pd.DataFrame: 日平均水位数据。

    Raises:
        ValueError: 当 ``instant_water_level_data`` 缺少必要列时。
    """
    required_cols = ["time", "water_level"]
    if not all(col in instant_water_level_data.columns for col in required_cols):
        raise ValueError(
            f"instant_water_level_data 必须包含 {required_cols} 列，"
            f"实际列: {instant_water_level_data.columns.tolist()}"
        )

    df = instant_water_level_data.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["date"] = df["time"].dt.date

    daily_records = []
    for date, group in df.groupby("date"):
        group = group.sort_values("time")

        water_levels = group["water_level"].values
        daily_amplitude = water_levels.max() - water_levels.min()

        if daily_amplitude < ARITHMETIC_DAILY_VARIATION_THRESHOLD and _is_equal_interval(
            group["time"]
        ):
            daily_water_level = water_levels.mean()
        else:
            daily_water_level = _area_bounded_mean(group)

        daily_records.append(
            {
                "time": pd.Timestamp(date),
                "time_GMT": pd.Timestamp(date),
                "water_level": daily_water_level,
                "unit": unit,
                "TResolution": "daily",
                "station_id": station_id,
            }
        )

    result = pd.DataFrame(daily_records)

    if daily_water_level_data is not None:
        result = pd.concat([result, daily_water_level_data], ignore_index=True)
        result = result.sort_values("time").reset_index(drop=True)

    return result


def _is_equal_interval(times: pd.Series) -> bool:
    """判断时间序列是否为等时距。"""
    if len(times) < 2:
        return True

    diffs = times.diff().dropna()
    if diffs.empty or diffs.nunique() == 1:
        return True

    return False


def _area_bounded_mean(group: pd.DataFrame) -> float:
    """使用面积包围法计算日平均水位。"""
    group = group.sort_values("time").reset_index(drop=True)
    times = group["time"]
    levels = group["water_level"]

    if len(group) == 1:
        return levels.iloc[0]

    day_start = times.min().replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + pd.Timedelta(hours=24)

    full_times = [day_start]
    full_levels = [levels.iloc[0]]

    for i, (t, l) in enumerate(zip(times, levels)):
        if t > day_start and t < day_end:
            full_times.append(t)
            full_levels.append(l)

    full_times.append(day_end)
    full_levels.append(levels.iloc[-1])

    full_series = pd.Series(full_levels, index=pd.DatetimeIndex(full_times))

    area = np.trapz(full_series.values, full_series.index.astype(np.int64) / 1e9)
    daily_mean = area / (24 * 3600)

    return daily_mean


def frequency(
    daily_water_level_data: pd.DataFrame,
    water_level_thresholds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算水位频率表和历时表。

    Args:
        daily_water_level_data: 日平均水位数据，需包含 ``time`` 和 ``water_level`` 列。
        water_level_thresholds: 水位阈值列表，用于划分水位等级。

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            1. frequency_df - 水位频率表
            2. duration_df - 水位历时表

    Raises:
        ValueError: 当 ``daily_water_level_data`` 缺少必要列时。
    """
    required_cols = ["time", "water_level"]
    if not all(col in daily_water_level_data.columns for col in required_cols):
        raise ValueError(
            f"daily_water_level_data 必须包含 {required_cols} 列，"
            f"实际列: {daily_water_level_data.columns.tolist()}"
        )

    df = daily_water_level_data.copy()
    total_days = len(df)

    if total_days == 0:
        empty_df = pd.DataFrame(
            {
                "water_level_threshold": water_level_thresholds,
                "days_count": [0] * len(water_level_thresholds),
                "percent": [0.0] * len(water_level_thresholds),
            }
        )
        return empty_df, empty_df.copy()

    frequency_records = []
    for threshold in water_level_thresholds:
        days_count = (df["water_level"] >= threshold).sum()
        percent = (days_count / total_days) * 100
        frequency_records.append(
            {
                "water_level_threshold": threshold,
                "days_count": days_count,
                "percent": round(percent, 2),
            }
        )

    frequency_df = pd.DataFrame(frequency_records)

    duration_records = []
    for threshold in water_level_thresholds:
        duration_days_count = (df["water_level"] < threshold).sum()
        duration_percent = (duration_days_count / total_days) * 100
        duration_records.append(
            {
                "water_level_threshold": threshold,
                "duration_days_count": duration_days_count,
                "duration_percent": round(duration_percent, 2),
            }
        )

    duration_df = pd.DataFrame(duration_records)

    return frequency_df, duration_df

