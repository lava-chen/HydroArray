"""水文模型相关命令"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from .output import success_result, error_result


def register(subparsers: argparse.ArgumentParser) -> None:
    """注册 model 子命令

    水文模型操作模块，包括：
    - list: 列出可用模型
    - create: 创建模型实例
    - run: 运行模型
    - calibrate: 模型率定
    """
    model_parser = subparsers.add_parser(
        "model",
        help="水文模型操作",
        description="水文模型的创建、运行、率定和评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用模型类型:
  水量平衡模型: xinanjiang, hymod, sac, crest
  河道演进模型: linear, kinematic, muskingum

示例:
  # 列出所有可用模型
  hydro model list

  # 运行新安江模型
  hydro model run -m xinanjiang -i input/ -o output/

  # 率定模型参数
  hydro model calibrate -m xinanjiang --observed obs.csv --method sce
        """
    )

    sub = model_parser.add_subparsers(dest="model_action", required=True,
                                      help="模型操作")

    # 模型列表
    _register_list(sub)

    # 创建模型
    _register_create(sub)

    # 运行模型
    _register_run(sub)

    # 模型率定
    _register_calibrate(sub)


def _register_list(sub) -> None:
    """注册模型列表子命令"""
    p = sub.add_parser(
        "list",
        help="列出可用模型",
        description="列出 HydroArray 支持的所有水文模型"
    )
    p.add_argument(
        "--category", "-c",
        choices=["water_balance", "routing", "all"],
        default="all",
        help="按类别筛选 (默认: all)"
    )
    p.add_argument(
        "--format", "-f",
        choices=["json", "text"],
        help="输出格式 (默认: 继承全局设置)"
    )
    p.set_defaults(func=_list_models)


def _register_create(sub) -> None:
    """注册模型创建子命令"""
    p = sub.add_parser(
        "create",
        help="创建水文模型实例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="创建配置好的水文模型实例",
        epilog="""
参数文件格式 (JSON):
  {
    "type": "xinanjiang",
    "parameters": {
      "K": 0.3,
      "IM": 0.01,
      "B": 0.5,
      ...
    },
    "routing": {
      "type": "muskingum",
      "parameters": {...}
    }
  }
        """
    )
    p.add_argument(
        "--type", "-t",
        required=True,
        choices=["xinanjiang", "hymod", "sac", "crest",
                "linear", "kinematic", "muskingum"],
        help="模型类型"
    )
    p.add_argument(
        "--config", "-c",
        help="模型配置文件路径 (JSON/YAML)"
    )
    p.add_argument(
        "--params", "-p",
        help="模型参数字符串 (JSON格式)"
    )
    p.add_argument(
        "--name", "-n",
        help="模型名称标签 (用于标识)"
    )
    p.set_defaults(func=_create_model)


def _register_run(sub) -> None:
    """注册模型运行子命令"""
    p = sub.add_parser(
        "run",
        help="运行水文模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="运行水文模型进行模拟",
        epilog="""
输入数据目录结构:
  input/
    precipitation.csv   # 降水
    temperature.csv     # 温度
    evaporation.csv     # 蒸发
    flow.csv            # 流量 (率定时使用)

CSV 文件格式:
  datetime,value
  2024-01-01,10.5
  2024-01-02,8.2

输出文件:
  <output>/runoff.csv       # 产流量
  <output>/streamflow.csv   # 河道流量
  <output>/states.json     # 模型状态 (如指定 --save-states)
        """
    )
    p.add_argument(
        "--model", "-m",
        required=True,
        help="模型名称、类型或配置文件路径"
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        help="输入数据目录或文件路径"
    )
    p.add_argument(
        "--output", "-o",
        help="输出目录路径 (默认: 输出到 stdout)"
    )
    p.add_argument(
        "--start-date", "-s",
        help="模拟起始日期 (YYYY-MM-DD)"
    )
    p.add_argument(
        "--end-date", "-e",
        help="模拟结束日期 (YYYY-MM-DD)"
    )
    p.add_argument(
        "--save-states",
        action="store_true",
        help="保存模型最终状态到 JSON"
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="显示运行进度到 stderr"
    )
    p.set_defaults(func=_run_model)


def _register_calibrate(sub) -> None:
    """注册模型率定子命令"""
    p = sub.add_parser(
        "calibrate",
        help="率定水文模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="使用优化算法率定水文模型参数",
        epilog="""
优化方法:
  - genetic: 遗传算法，适合全局优化
  - sce: SCE-UA (Shuffled Complex Evolution)，推荐用于水文模型
  - scipy: Scipy 优化器，支持多种算法

目标函数:
  - nse: Nash-Sutcliffe Efficiency (默认)
  - kge: Kling-Global Efficiency
  - rmse: Root Mean Square Error
  - mae: Mean Absolute Error

率定结果输出:
  {
    "optimal_parameters": {...},
    "objective_value": 0.856,
    "iterations": 500,
    "evaluations": 2500
  }

使用 --output 保存到文件，或用 > 重定向到 stdout
        """
    )
    p.add_argument(
        "--model", "-m",
        required=True,
        help="模型名称或配置文件路径"
    )
    p.add_argument(
        "--observed", "-obs",
        required=True,
        help="实测流量数据文件 (CSV)"
    )
    p.add_argument(
        "--method",
        choices=["genetic", "sce", "scipy"],
        default="sce",
        help="优化方法 (默认: sce)"
    )
    p.add_argument(
        "--objective", "-obj",
        choices=["nse", "kge", "rmse", "mae"],
        default="nse",
        help="目标函数 (默认: nse)"
    )
    p.add_argument(
        "--iterations", "-n",
        type=int,
        default=1000,
        help="最大迭代次数 (默认: 1000)"
    )
    p.add_argument(
        "--output", "-o",
        help="率定结果输出文件 (JSON)"
    )
    p.add_argument(
        "--save-best",
        action="store_true",
        help="保存率定后的模型配置"
    )
    p.set_defaults(func=_calibrate_model)


def _list_models(args) -> Dict:
    """列出可用模型"""
    from HydroArray.domain.models import get_available_models

    models = get_available_models()

    if args.category != "all":
        models = {args.category: models.get(args.category, [])}

    return success_result(
        data=models,
        meta={"count": len(models)}
    )


def _create_model(args) -> Dict:
    """创建模型"""
    from HydroArray.domain.models import create_model, create_hydrology_model

    config = {}

    if args.config:
        with open(args.config) as f:
            if args.config.endswith((".yaml", ".yml")):
                import yaml
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

    if args.params:
        params = json.loads(args.params)
        config.update(params)

    model_type = args.type.lower()
    if model_type in ["xinanjiang", "hymod", "sac", "crest"]:
        model = create_hydrology_model(model_type, **config)
    else:
        model = create_model(model_type, **config)

    result_data = {
        "model_type": args.type,
        "name": args.name or model_type,
        "parameters": model.parameters if hasattr(model, 'parameters') else {},
    }

    return success_result(
        data=result_data,
        message=f"模型 {args.type} 创建成功"
    )


def _run_model(args) -> Dict:
    """运行模型"""
    from HydroArray.domain.models import create_hydrology_model
    import pandas as pd
    from pathlib import Path

    if args.model.endswith(('.json', '.yaml', '.yml')):
        with open(args.model) as f:
            if args.model.endswith(('.yaml', '.yml')):
                import yaml
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        model = create_hydrology_model(config.get("type", "xinanjiang"), **config)
    else:
        model = create_hydrology_model(args.model)

    input_path = Path(args.input)
    forcings = {}

    if input_path.is_dir():
        for csv_file in input_path.glob("*.csv"):
            df = pd.read_csv(csv_file)
            name = csv_file.stem
            forcings[name] = df
    else:
        df = pd.read_csv(args.input)
        forcings["default"] = df

    result = model.run(forcings)

    output_data = {
        "runoff": result.get("runoff", []),
        "streamflow": result.get("streamflow", []),
    }

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, data in output_data.items():
            if isinstance(data, pd.DataFrame):
                data.to_csv(output_dir / f"{name}.csv", index=False)
            elif isinstance(data, (list, dict)):
                pd.DataFrame(data).to_csv(output_dir / f"{name}.csv", index=False)
        if args.save_states:
            import json
            with open(output_dir / "states.json", 'w') as f:
                json.dump(result.get("states", {}), f)

    return success_result(
        data=output_data,
        meta={
            "model": args.model,
            "input": str(input_path),
            "n_timesteps": len(output_data.get("runoff", []))
        }
    )


def _calibrate_model(args) -> Dict:
    """率定模型"""
    from HydroArray.domain.calibration import calibrate, SCEOptimizer, GeneticAlgorithm
    from HydroArray.domain.models import create_hydrology_model
    from HydroArray.utils.metrics import evaluate_model
    import pandas as pd

    obs_df = pd.read_csv(args.observed)
    observed = obs_df.iloc[:, 0].values

    with open(args.model) as f:
        if args.model.endswith(('.yaml', '.yml')):
            import yaml
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    model = create_hydrology_model(config.get("type", "xinanjiang"), **config)

    objective_map = {
        "nse": lambda o, s: -evaluate_model(o, s, "nse")["nse"],
        "kge": lambda o, s: -evaluate_model(o, s, "kge")["kge"],
        "rmse": lambda o, s: evaluate_model(o, s, "rmse")["rmse"],
        "mae": lambda o, s: evaluate_model(o, s, "mae")["mae"],
    }

    objective_func = objective_map[args.objective]

    if args.method == "sce":
        optimizer = SCEOptimizer(max_iterations=args.iterations)
    else:
        optimizer = GeneticAlgorithm(max_iterations=args.iterations)

    result = calibrate(model, observed, objective_func, optimizer)

    output = {
        "optimal_parameters": result.parameters if hasattr(result, 'parameters') else {},
        "objective_value": float(result.objective_value) if hasattr(result, 'objective_value') else None,
        "iterations": result.iterations if hasattr(result, 'iterations') else args.iterations,
        "evaluations": result.evaluations if hasattr(result, 'evaluations') else None,
    }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)

    if args.save_best:
        output_path = Path(args.output).parent if args.output else Path.cwd()
        with open(output_path / f"calibrated_{config.get('type', 'model')}.json", 'w') as f:
            json.dump({**config, "calibrated_parameters": output["optimal_parameters"]}, f, indent=2)

    return success_result(
        data=output,
        message=f"率定完成，目标函数值: {output['objective_value']:.6f}"
    )