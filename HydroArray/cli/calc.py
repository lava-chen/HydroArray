"""水文计算相关命令"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

from .output import OutputFormatter, success_result, format_metric_result


def register(subparsers: argparse.ArgumentParser) -> None:
    """注册 calc 子命令

    水文过程计算模块，包括：
    - runoff: 产流/径流计算（饱和超额、双源、三源模型）
    - evaporation: 蒸散发计算（三层蒸发、Penman 模型）
    - cross-section: 河道横截面计算（面积、湿周、水力半径）
    """
    calc_parser = subparsers.add_parser(
        "calc",
        help="水文过程计算",
        description="产流、蒸发、截面等水文过程计算",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 饱和超额产流计算
  hydro calc runoff -t saturation -p precip.csv -sm soil.csv

  # 双源蒸散发分离
  hydro calc evaporation -t three-layer -temp temp.csv -hum 0.7

  # 河道横截面面积计算
  hydro calc cross-section -d cross_section.csv --calculate-area
        """
    )

    sub = calc_parser.add_subparsers(dest="calc_type", required=True,
                                     help="计算类型")

    # 产流计算
    _register_runoff(sub)

    # 蒸发计算
    _register_evaporation(sub)

    # 截面计算
    _register_cross_section(sub)


def _register_runoff(sub) -> None:
    """注册产流计算子命令

    产流模型说明:
    - saturation: 饱和超额产流模型，适合湿润地区
    - two-source: 双源径流分离，考虑地表径流和壤中流
    - three-source: 三源径流分离，考虑地表径流、壤中流和基流
    """
    p = sub.add_parser(
        "runoff",
        help="径流/产流计算",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="计算流域产流量",
        epilog="""
输入数据格式 (CSV):
  datetime,precipitation,soil_moisture
  2024-01-01,10.5,0.35
  2024-01-02,8.2,0.38

必需列: datetime, precipitation
可选列: soil_moisture, temperature, evapotranspiration
        """
    )

    p.add_argument(
        "--type", "-t",
        choices=["saturation", "two-source", "three-source"],
        default="saturation",
        help="产流模型类型 (默认: saturation)"
    )
    p.add_argument(
        "--precipitation", "-p",
        required=True,
        help="降水量数据文件路径 (CSV格式，需包含 precipitation 列)"
    )
    p.add_argument(
        "--soil-moisture", "-sm",
        help="土壤含水量数据文件路径 (CSV格式，需包含 soil_moisture 列)"
    )
    p.add_argument(
        "--vegetation-cover", "-vc",
        type=float,
        help="植被覆盖度，范围 0-1 (可选)"
    )
    p.add_argument(
        "--initial-storage", "-is",
        type=float,
        help="初始蓄水量 (可选)"
    )
    p.add_argument(
        "--output", "-o",
        choices=["json", "text", "compact"],
        default=None,
        help="输出格式 (默认: 继承全局设置)"
    )
    p.set_defaults(func=_run_runoff)


def _register_evaporation(sub) -> None:
    """注册蒸发计算子命令

    蒸发模型说明:
    - three-layer: 三层蒸发模型，考虑土壤表层、中层、深层蒸发
    - penman: Penman-Monteith 公式，需要温度、湿度、风速、辐射数据
    """
    p = sub.add_parser(
        "evaporation",
        help="蒸散发计算",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="计算流域蒸散发量",
        epilog="""
输入数据格式 (CSV):
  datetime,temperature,humidity,wind_speed,radiation
  2024-01-01,25.5,0.65,2.3,15.2
  2024-01-02,26.0,0.60,2.5,14.8

必需列: datetime, temperature
可选列: humidity (0-1), wind_speed (m/s), radiation (MJ/m2/day)
        """
    )

    p.add_argument(
        "--type", "-t",
        choices=["three-layer", "penman"],
        default="three-layer",
        help="蒸发模型类型 (默认: three-layer)"
    )
    p.add_argument(
        "--temperature", "-temp",
        required=True,
        help="温度数据文件路径 (CSV格式，单位: 摄氏度)"
    )
    p.add_argument(
        "--humidity", "-hum",
        type=float,
        help="相对湿度，范围 0-1"
    )
    p.add_argument(
        "--wind-speed", "-wind",
        type=float,
        help="风速 (单位: m/s)"
    )
    p.add_argument(
        "--radiation", "-rad",
        help="辐射数据文件路径 (CSV格式，单位: MJ/m²/day)"
    )
    p.add_argument(
        "--land-use", "-land",
        choices=["forest", "grassland", "cropland", "bare", "water"],
        help="土地利用类型"
    )
    p.add_argument(
        "--latent-heat", "-lh",
        type=float,
        default=2.45,
        help="汽化潜热 MJ/kg (默认: 2.45)"
    )
    p.add_argument(
        "--output", "-o",
        choices=["json", "text", "compact"],
        default=None,
        help="输出格式 (默认: 继承全局设置)"
    )
    p.set_defaults(func=_run_evaporation)


def _register_cross_section(sub) -> None:
    """注册横截面计算子命令

    计算河道横截面几何特征：
    - 面积: 过水断面面积
    - 湿周: 水流与河床接触的周长
    - 水力半径: 面积/湿周
    - 曼宁系数: 根据河床特性确定
    """
    p = sub.add_parser(
        "cross-section",
        help="河道横截面计算",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="计算河道横截面几何特征",
        epilog="""
输入数据格式 (CSV):
  distance,elevation
  0.0,100.0
  5.0,98.5
  10.0,95.0
  ...

必需列: distance (距离, m), elevation (高程, m)
可选列: roughness (糙率), width (水面宽)

计算结果包含:
  - cross_section_area: 过水面积 (m²)
  - wetted_perimeter: 湿周 (m)
  - hydraulic_radius: 水力半径 (m)
  - max_depth: 最大水深 (m)
        """
    )

    p.add_argument(
        "--data", "-d",
        required=True,
        help="横断面测量数据文件路径 (CSV格式)"
    )
    p.add_argument(
        "--calculate-area",
        action="store_true",
        help="计算过水面积"
    )
    p.add_argument(
        "--calculate-wetted-perimeter",
        action="store_true",
        help="计算湿周"
    )
    p.add_argument(
        "--water-level", "-wl",
        type=float,
        help="水位高程 (m)，用于计算特定水位下的过水面积"
    )
    p.add_argument(
        "--manning-n", "-n",
        type=float,
        help="曼宁糙率系数"
    )
    p.add_argument(
        "--output", "-o",
        choices=["json", "text", "compact"],
        default=None,
        help="输出格式 (默认: 继承全局设置)"
    )
    p.set_defaults(func=_run_cross_section)


def _load_csv_data(path: str) -> Any:
    """加载 CSV 数据文件"""
    import pandas as pd
    from pathlib import Path

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件未找到: {path}")

    return pd.read_csv(path)


def _run_runoff(args) -> Dict:
    """执行产流计算"""
    from HydroArray.domain.process import (
        saturation_excess_runoff,
        two_source_runoff_separation,
        three_source_runoff_separation,
    )

    precip_df = _load_csv_data(args.precipitation)

    # 提取必要列
    precip = precip_df["precipitation"].values

    kwargs = {}
    if args.soil_moisture:
        soil_df = _load_csv_data(args.soil_moisture)
        if "soil_moisture" in soil_df.columns:
            kwargs["soil_moisture"] = soil_df["soil_moisture"].values
    if args.vegetation_cover is not None:
        kwargs["vegetation_cover"] = args.vegetation_cover
    if args.initial_storage is not None:
        kwargs["initial_storage"] = args.initial_storage

    if args.type == "saturation":
        result = saturation_excess_runoff(precip, **kwargs)
    elif args.type == "two-source":
        result = two_source_runoff_separation(precip)
    else:  # three-source
        result = three_source_runoff_separation(precip)

    return success_result(
        data=result,
        meta={
            "model_type": args.type,
            "input_file": args.precipitation,
            "n_timesteps": len(precip)
        }
    )


def _run_evaporation(args) -> Dict:
    """执行蒸发计算"""
    from HydroArray.domain.process import three_layer_evaporation

    temp_df = _load_csv_data(args.temperature)
    temp = temp_df["temperature"].values

    kwargs = {}
    if args.humidity is not None:
        kwargs["humidity"] = args.humidity
    if args.wind_speed is not None:
        kwargs["wind_speed"] = args.wind_speed
    if args.radiation:
        rad_df = _load_csv_data(args.radiation)
        if "radiation" in rad_df.columns:
            kwargs["radiation"] = rad_df["radiation"].values
    if args.land_use:
        kwargs["land_use"] = args.land_use
    if args.latent_heat:
        kwargs["latent_heat"] = args.latent_heat

    result = three_layer_evaporation(temperature=temp, **kwargs)

    return success_result(
        data=result,
        meta={
            "model_type": args.type,
            "input_file": args.temperature,
            "n_timesteps": len(temp)
        }
    )


def _run_cross_section(args) -> Dict:
    """执行横截面计算"""
    from HydroArray.domain.process import (
        calculate_channel_section_detailed,
        calculate_cross_section_area,
    )

    data = _load_csv_data(args.data)

    if args.water_level is not None:
        # 计算特定水位下的面积
        result = calculate_cross_section_area(data, water_level=args.water_level)
    elif args.calculate_area:
        result = calculate_cross_section_area(data)
    else:
        result = calculate_channel_section_detailed(data)

    return success_result(
        data=result,
        meta={
            "input_file": args.data,
            "n_points": len(data)
        }
    )