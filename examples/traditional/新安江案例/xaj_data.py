"""
新安江模型案例 · 数据加载
========================
读取日资料、小时资料、子流域配置、场次配置。
面雨量按子流域面积加权; 流量 m³/s → mm/步。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml

import HydroArray as ha

from xaj_config import (
    BASE, AREA_KM2, STEP_H_DAILY, STEP_H_EVENT,
    load_ref_params, build_initial_params,
)


@dataclass
class DailyData:
    """日模型输入数据包。"""
    df: pd.DataFrame              # 原始 DataFrame (含 q_obs, evap, rain_*)
    precip: pd.Series             # 面雨量 (mm/d)
    precip_subbasin: pd.DataFrame # 10 个子流域雨量 (mm/d)
    pet: pd.Series                # 面蒸发 (mm/d, 蒸发皿)
    q_obs_mm: pd.Series           # 实测径流深 (mm/d)
    ref: dict                     # 参数参考表
    events: List[dict]            # 次洪场次配置


@dataclass
class HourlyData:
    """次洪模型输入数据包。"""
    df: pd.DataFrame              # 原始 DataFrame (含 q_obs, rain_*)
    precip_h: pd.Series           # 面雨量 (mm/h)
    precip_subbasin_h: pd.DataFrame # 10 个子流域雨量 (mm/h)
    pet_h: pd.Series              # 面蒸发 (mm/h, 日值/24 等分)


def load_subbasins() -> list[dict]:
    """读取 subbasins.yaml。"""
    with open(BASE / "subbasins.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["subbasins"]


def _load_subbasin_weights() -> np.ndarray:
    """读取 subbasins.yaml, 返回面积权重数组。"""
    cfg = load_subbasins()
    return np.array([rec["area_ratio"] for rec in cfg])


def _compute_subbasin_precip(df: pd.DataFrame) -> pd.DataFrame:
    """按 subbasins.yaml 的站点顺序返回各子流域雨量矩阵。"""
    cfg = load_subbasins()
    cols = [f"rain_{rec['station']}" for rec in cfg]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少子流域雨量列: {missing}")
    out = df[cols].copy()
    out.columns = [rec["station"] for rec in cfg]
    return out


def _compute_area_precip(df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """面积加权计算面雨量。"""
    rain_cols = [c for c in df.columns if c.startswith("rain_")]
    if len(rain_cols) != len(weights):
        raise ValueError(
            f"雨量站数 ({len(rain_cols)}) 与子流域权重数 ({len(weights)}) 不匹配"
        )
    return pd.Series((df[rain_cols].values * weights).sum(axis=1),
                     index=df.index, name="precip")


def load_daily() -> DailyData:
    """加载日资料 + 配置, 返回 DailyData 数据包。"""
    df = pd.read_csv(BASE / "xaj_daily.csv",
                     parse_dates=["time"], encoding="utf-8-sig")
    df = df.set_index("time").sort_index()

    weights = _load_subbasin_weights()
    precip_subbasin = _compute_subbasin_precip(df)
    precip = _compute_area_precip(df, weights)
    pet = df["evap"].copy()
    pet.name = "pet"
    # q_obs: m³/s → mm/d
    q_obs_mm = pd.Series(
        ha.cms_to_mm(df["q_obs"].values, area=AREA_KM2, dt_hours=STEP_H_DAILY),
        index=df.index, name="q_obs_mm"
    )

    ref = load_ref_params()
    with open(BASE / "events_event.yaml", encoding="utf-8") as f:
        events = yaml.safe_load(f)["events_event"]

    return DailyData(df=df, precip=precip, precip_subbasin=precip_subbasin, pet=pet,
                     q_obs_mm=q_obs_mm, ref=ref, events=events)


def _diurnal_pet_weights(idx: pd.DatetimeIndex) -> np.ndarray:
    """构造按日期下推到小时的日蒸发权重: 白天半正弦分配。

    白天定义为 8:00-20:00 (12 h), 夜间 (20:00-次日 8:00) 权重 0。
    白天 12 h 按半正弦曲线 sin(πτ/T) 加权, 每日单独归一化, 使日总量保持不变。
    比简单 24 等分更符合实际蒸散发的昼夜差异。
    """
    hours = idx.hour.values
    daylight = (hours >= 8) & (hours < 20)
    w = np.zeros(len(idx), dtype=float)
    tau = (hours[daylight] - 8).astype(float)
    w[daylight] = np.sin(np.pi * tau / 12.0)
    s = pd.Series(w).groupby(idx.normalize()).transform("sum").values
    s[s <= 0.0] = 1.0
    return w / s


def load_hourly(daily_df: pd.DataFrame) -> HourlyData:
    """加载小时资料; 日蒸发按"白天半正弦"分配到小时。

    白天定义为 8:00-20:00 (12 h), 夜间权重 0;
    白天按半正弦曲线加权并归一化, 日总量保持不变。
    """
    df = pd.read_csv(BASE / "xaj_hourly.csv",
                     parse_dates=["time"], encoding="utf-8-sig")
    df = df.set_index("time").sort_index()

    weights = _load_subbasin_weights()
    precip_subbasin_h = _compute_subbasin_precip(df)
    precip_h = _compute_area_precip(df, weights)

    daily_pet = daily_df["evap"]
    pet_per_day = daily_pet.reindex(df.index.normalize()).values
    diurnal_w = _diurnal_pet_weights(df.index)
    pet_h = pd.Series(pet_per_day * diurnal_w,
                      index=df.index, name="pet")

    return HourlyData(df=df, precip_h=precip_h,
                      precip_subbasin_h=precip_subbasin_h, pet_h=pet_h)


def slice_event(hourly: HourlyData, t_start: str, t_end: str,
                warmup_h: int = 24) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray,
                                              pd.DatetimeIndex, int]:
    """切出单场洪水时段 (含 warmup_h 预热)。

    Returns:
        (precip, pet, q_obs_cms, q_obs_mm_h, time_index, warmup_h)
    """
    ts = pd.Timestamp(t_start)
    te = pd.Timestamp(t_end)
    warm_start = ts - pd.Timedelta(hours=warmup_h)

    segment = hourly.df.loc[warm_start:te]
    if segment.empty:
        raise ValueError(f"场次 {t_start} ~ {t_end} 没有小时资料")

    # 小时资料只覆盖部分洪水过程，不能假定每场都有完整 warmup_h。
    # 返回实际存在的事件前样本数，避免误删洪水起涨段。
    actual_warmup = int((segment.index < ts).sum())
    p_in = hourly.precip_subbasin_h.reindex(segment.index).values
    e_in = hourly.pet_h.reindex(segment.index).values
    q_obs_cms = segment["q_obs"].values.astype(float)
    q_obs_mm_h = ha.cms_to_mm(q_obs_cms, area=AREA_KM2, dt_hours=STEP_H_EVENT)
    t_idx = segment.index.copy()

    return p_in, e_in, q_obs_cms, q_obs_mm_h, t_idx, actual_warmup
