"""水文评价指标相关命令"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from .output import success_result, format_metric_result


def register(subparsers: argparse.ArgumentParser) -> None:
    """注册 metrics 子命令

    水文评价指标计算模块，包括：
    - calculate: 计算单个或多个指标
    - batch: 批量计算多个指标
    - compare: 比较两组数据
    """
    metrics_parser = subparsers.add_parser(
        "metrics",
        help="水文评价指标计算",
        description="NSE、KGE、RMSE 等模型评价指标计算",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用指标:
  nse      - Nash-Sutcliffe Efficiency (纳什效率系数)
  kge     - Kling-Global Efficiency (克林效率系数)
  rmse    - Root Mean Square Error (均方根误差)
  mae     - Mean Absolute Error (平均绝对误差)
  pbias   - Percent Bias (百分比偏差)
  bias    - Bias (偏差)
  r2      - R-squared (决定系数)
  nse_log - Logarithmic NSE (对数NSE)

输入格式:
  CSV 文件路径或逗号分隔的数值列表

示例:
  # 计算单个指标
  hydro metrics nse --observed obs.csv --simulated sim.csv

  # 批量计算多个指标
  hydro metrics batch --observed obs.csv --simulated sim.csv -m nse kge rmse

  # 直接使用数值
  hydro metrics nse --observed "1,2,3,4,5" --simulated "1.1,2.2,3.1,4.2,4.9"
        """
    )

    sub = metrics_parser.add_subparsers(dest="metric_type", required=True,
                                        help="操作类型")

    # 单指标计算
    _register_single_metric(sub)

    # 批量计算
    _register_batch(sub)


def _register_single_metric(sub) -> None:
    """注册单指标计算子命令"""
    metric_names = {
        "nse": "Nash-Sutcliffe Efficiency",
        "kge": "Kling-Global Efficiency",
        "rmse": "Root Mean Square Error",
        "mae": "Mean Absolute Error",
        "pbias": "Percent Bias",
        "bias": "Bias",
        "r2": "R-squared (决定系数)",
        "nse_log": "Logarithmic NSE",
    }

    for metric_name, metric_desc in metric_names.items():
        metric_parser = sub.add_parser(
            metric_name,
            help=f"计算 {metric_desc}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=metric_desc
        )
        metric_parser.add_argument(
            "--observed", "-obs",
            required=True,
            help="实测数据 (CSV文件路径或逗号分隔的数值)"
        )
        metric_parser.add_argument(
            "--simulated", "-sim",
            required=True,
            help="模拟数据 (CSV文件路径或逗号分隔的数值)"
        )
        metric_parser.add_argument(
            "--unit",
            help="指标单位 (可选)"
        )
        metric_parser.add_argument(
            "--output", "-o",
            choices=["json", "text", "compact"],
            help="输出格式 (默认: 继承全局设置)"
        )
        metric_parser.set_defaults(func=lambda a, m=metric_name: _calculate_metric(a, m))


def _register_batch(sub) -> None:
    """注册批量计算子命令"""
    p = sub.add_parser(
        "batch",
        help="批量计算多个指标",
        description="一次计算多个评价指标"
    )
    p.add_argument(
        "--observed", "-obs",
        required=True,
        help="实测数据"
    )
    p.add_argument(
        "--simulated", "-sim",
        required=True,
        help="模拟数据"
    )
    p.add_argument(
        "--metrics", "-m",
        nargs="+",
        default=["nse", "kge", "rmse", "mae", "pbias"],
        help="要计算的指标列表 (默认: nse kge rmse mae pbias)"
    )
    p.add_argument(
        "--output", "-o",
        choices=["json", "text", "compact"],
        help="输出格式 (默认: 继承全局设置)"
    )
    p.set_defaults(func=_batch_calculate)


def _parse_values(value: Any) -> Any:
    """解析数值，支持文件路径或逗号分隔的字符串"""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    if isinstance(value, str):
        # 如果是文件路径
        if Path(value).exists() and (value.endswith('.csv') or ',' not in value):
            try:
                df = pd.read_csv(value)
                return df.iloc[:, 0].values
            except:
                pass

        # 如果是逗号分隔的字符串
        if ',' in value:
            try:
                return np.array([float(x.strip()) for x in value.split(',')])
            except ValueError:
                pass

        # 尝试 JSON 解析
        try:
            import json
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return np.array(parsed)
        except:
            pass

    return value


def _calculate_metric(args, metric_name: str) -> Dict:
    """计算单个指标"""
    from HydroArray.utils.metrics import (
        calculate_nse, calculate_kge, calculate_rmse, calculate_mae,
        calculate_pbias, calculate_bias, calculate_r_squared, calculate_nse_log
    )

    metric_funcs = {
        "nse": (calculate_nse, ""),
        "kge": (calculate_kge, ""),
        "rmse": (calculate_rmse, ""),
        "mae": (calculate_mae, ""),
        "pbias": (calculate_pbias, "%"),
        "bias": (calculate_bias, ""),
        "r2": (calculate_r_squared, ""),
        "nse_log": (calculate_nse_log, ""),
    }

    obs = _parse_values(args.observed)
    sim = _parse_values(args.simulated)

    if hasattr(obs, 'values'):
        obs = obs.values
    if hasattr(sim, 'values'):
        sim = sim.values

    func, unit = metric_funcs[metric_name]
    metric_value = func(obs, sim)

    return success_result(
        data=format_metric_result(
            name=metric_name.upper(),
            value=metric_value,
            unit=unit or args.unit
        ),
        meta={
            "n": len(obs) if hasattr(obs, '__len__') else "unknown",
            "metric_type": metric_name
        }
    )


def _batch_calculate(args) -> Dict:
    """批量计算指标"""
    from HydroArray.utils.metrics import evaluate_model

    obs = _parse_values(args.observed)
    sim = _parse_values(args.simulated)

    if hasattr(obs, 'values'):
        obs = obs.values
    if hasattr(sim, 'values'):
        sim = sim.values

    results = {}
    for metric in args.metrics:
        metric_lower = metric.lower()
        if metric_lower in ["nse", "kge", "rmse", "mae", "pbias", "bias", "r2", "nse_log"]:
            result = evaluate_model(obs, sim, metric_lower)
            if isinstance(result, dict):
                # 提取单个值
                value_key = metric_lower if metric_lower != "nse_log" else "nse_log"
                results[metric] = {k: float(v) if isinstance(v, (int, float)) else v
                                   for k, v in result.items()}
            else:
                results[metric] = float(result)
        else:
            results[metric] = {"error": f"未知指标: {metric}"}

    return success_result(
        data={
            "metrics": results,
            "count": len(args.metrics),
            "n": len(obs) if hasattr(obs, '__len__') else "unknown"
        }
    )