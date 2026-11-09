"""
新安江模型案例 · 全局配置与常量
================================
集中管理路径、流域参数、时间步长、SL 250 许可误差、
率定/验证时段划分、参数边界等。
所有模块均从此处导入常量, 避免魔法数字散落。
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import yaml

import HydroArray as ha
from HydroArray.domain import ParameterBounds


# ─────────────────── 路径 ───────────────────
BASE = Path(__file__).parent
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)

# ─────────────────── 流域与步长 ───────────────────
AREA_KM2 = 290.0          # 呈村流域控制面积 (km²)
STEP_H_DAILY = 24.0       # 日模型步长 (h)
STEP_H_EVENT = 1.0        # 次洪模型步长 (h)

# ─────────────────── SL 250《水文情报预报规范》许可误差 ───────────────────
PERMIT_DAILY_RUNOFF_PCT = 20.0    # 日径流深许可相对误差 (%)
PERMIT_EVENT_PEAK_PCT = 20.0      # 次洪洪峰许可相对误差 (%)
PERMIT_EVENT_VOL_PCT = 20.0       # 次洪洪量许可相对误差 (%)
PERMIT_EVENT_PEAK_LAG_H = 3.0     # 次洪峰时许可绝对误差 (h)

# ─────────────────── 率定/验证/预热时段划分 ───────────────────
# 1989 全年作为 warm-up, 不参与指标计算
WARMUP_END = "1989-12-31"          # 预热期截止 (含)
# 1990-1994 率定期, 1995-1996 验证期
CALIB_START = "1990-01-01"
CALIB_END = "1994-12-31"
VALID_START = "1995-01-01"
VALID_END = "1996-12-31"


# ═══════════════════════════════════════════════════════════════
# 固定参数 (FIXED) — 优化器绝对不动, 由教师/规范给定
# ═══════════════════════════════════════════════════════════════
# 物理量纲、单值常识, 没必要率定, 率定反而过拟合。
# 修改这里就等于"改答案", 不要从代码里改率定键名。
FIXED_PARAMS: Dict[str, float] = {
    "B":   0.3,    # 流域蓄水容量分布曲线指数 (长江委推荐 0.3)
    "WUM": 20.0,   # 上层张力水容量 mm (植被截留+表层)
    "WLM": 60.0,   # 下层张力水容量 mm
    "WDM": 40.0,   # 深层张力水容量 mm  (= WM - WUM - WLM, WM=120)
    "WM":  120.0,  # 张力水总容量 mm  (= WUM + WLM + WDM)
    "C":   0.18,   # 深层散发系数
    "IMP": 0.01,   # 不透水面积比例
    "EX":  1.5,    # 自由水容量分布曲线指数
    "CS":  0.5,    # 河网水流消退系数 (单节点不显式, 留接口)
    "KE":  1.0,    # Muskingum K 默认值 (h); 日模型应取 STEP_H_DAILY, 次洪取 STEP_H_EVENT
}

# ─────────────────── Muskingum 时间参数 K (按步长区分) ───────────────────
# 物理上, Muskingum K 应等于 (或近似于) 计算步长 dt, 这样参数有明确量纲含义。
# 日模型 dt=24h → KE=24h; 次洪 dt=1h → KE=1h。统一在 calibration 流程里按步长覆写。
KE_DAILY: float = STEP_H_DAILY     # = 24.0 h
KE_EVENT: float = STEP_H_EVENT     # = 1.0 h

# ═══════════════════════════════════════════════════════════════
# 率定参数 (CALIBRATED) — DE 唯一允许动的
# ═══════════════════════════════════════════════════════════════
# 给出"参考范围"和"硬约束边界"两套:
#   REF  : 教师/规范推荐范围 (报告里做"是否在范围内"判定)
#   BOUND: DE 实际搜索的硬边界 (放宽一些, 避免在边界上截断)

# ── 日模型率定 (24h 步长) ──
# K (蒸发折算) + SM (自由水) + KI/KG (分水源) + CI/CG (退水) = 6 个
DAILY_CALIB_KEYS: List[str] = ["K", "SM", "KI", "KG", "CI", "CG"]

DAILY_CALIB_REF: Dict[str, Tuple[float, float]] = {
    "K":  (0.8, 1.2),
    "SM": (10.0, 30.0),
    "KI": (0.1, 0.69),
    "KG": (0.01, 0.69),
    "CI": (0.1, 0.9),
    "CG": (0.98, 0.998),
}

DAILY_CALIB_BOUND: Dict[str, Tuple[float, float]] = {
    "K":  (0.6, 1.4),
    "SM": (5.0, 50.0),
    "KI": (0.01, 0.69),
    "KG": (0.01, 0.69),
    "CI": (0.01, 0.90),
    "CG": (0.90, 0.998),
}

# ── 次洪模型率定 (1h 步长) ──
# 日 6 个 + 河网滞时 L + Muskingum xe = 8 个
EVENT_CALIB_KEYS: List[str] = ["K", "SM", "KI", "KG", "CI", "CG", "L", "xe"]

EVENT_CALIB_REF: Dict[str, Tuple[float, float]] = {
    "K":  (0.8, 1.2),
    "SM": (10.0, 30.0),
    "KI": (0.1, 0.69),
    "KG": (0.01, 0.69),
    "CI": (0.1, 0.9),
    "CG": (0.98, 0.998),
    "L":  (0.0, 2.0),
    "xe": (0.30, 0.35),
}

EVENT_CALIB_BOUND: Dict[str, Tuple[float, float]] = {
    "K":  (0.6, 1.4),
    "SM": (5.0, 50.0),
    "KI": (0.01, 0.69),
    "KG": (0.01, 0.69),
    "CI": (0.01, 0.90),
    "CG": (0.90, 0.998),
    "L":  (0.0, 3.0),
    "xe": (0.0, 0.45),
}

# ── 合并参考范围 (供 report/word 模块使用, 判断率定值是否在规范范围内) ──
# 兼容旧 API: REF_RANGES, ref_value 在 ref 里的参考范围
REF_RANGES: Dict[str, Tuple[float, float]] = {
    **DAILY_CALIB_REF,
    "L":  (0.0, 2.0),
    "xe": (0.30, 0.35),
}

# 兼容旧模块导入。实际搜索键由 DAILY_CALIB_KEYS/EVENT_CALIB_KEYS 控制。
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    **DAILY_CALIB_BOUND,
    **EVENT_CALIB_BOUND,
    "KE": (FIXED_PARAMS["KE"], FIXED_PARAMS["KE"]),
}


# 参数参考值默认值 (当 params_ref.yaml 缺失时兜底)
PARAM_DEFAULTS: Dict[str, float] = {
    "K": 0.9, "B": 0.3, "C": 0.18, "IMP": 0.01,
    "WM": 120.0, "WUM": 20.0, "WLM": 60.0, "WDM": 40.0,
    "SM": 20.0, "EX": 1.5, "KI": 0.4, "KG": 0.1,
    "CI": 0.5, "CG": 0.99, "CS": 0.3, "L": 0.0,
    "KE": 1.0, "xe": 0.2,
}


def load_ref_params() -> Dict[str, str]:
    """读取 params_ref.yaml, 返回 {name: ref_value_str} 字典。"""
    with open(BASE / "params_ref.yaml", encoding="utf-8") as f:
        ref_list = yaml.safe_load(f)["params"]
    return {p["name"]: p["ref_value"] for p in ref_list}


def parse_ref_value(ref: Dict, key: str, default: float) -> float:
    """从 ref 字典里取标量, 支持 "0.3" / "0.8~1.2" / None 格式。"""
    # 兼容 IM/IMP 别名
    lookup_key = key
    if key == "IMP" and "IMP" not in ref and "IM" in ref:
        lookup_key = "IM"
    # 兼容 X/xe 别名
    if key == "xe" and "xe" not in ref and "X" in ref:
        lookup_key = "X"

    v = ref.get(lookup_key, default)
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if "~" in s:
        lo, hi = s.split("~", 1)
        return (float(lo) + float(hi)) / 2
    if s.startswith("0.7-"):  # "0.7-KG" 特殊处理
        return 0.5
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def build_initial_params(ref: Dict[str, str]) -> ha.XinAnjiangParameters:
    """根据 ref 表构造初始参数 (中值或默认)。"""
    p = ha.XinAnjiangParameters(
        K=parse_ref_value(ref, "K", 0.9),
        B=parse_ref_value(ref, "B", 0.3),
        IMP=parse_ref_value(ref, "IMP", 0.01),
        WM=parse_ref_value(ref, "WM", 120.0),
        WUM=parse_ref_value(ref, "WUM", 20.0),
        WLM=parse_ref_value(ref, "WLM", 60.0),
        EX=parse_ref_value(ref, "EX", 1.5),
        C=parse_ref_value(ref, "C", 0.18),
        SM=parse_ref_value(ref, "SM", 20.0),
        KI=parse_ref_value(ref, "KI", 0.4),
        KG=parse_ref_value(ref, "KG", 0.1),
        CI=parse_ref_value(ref, "CI", 0.5),
        CG=parse_ref_value(ref, "CG", 0.99),
        CS=parse_ref_value(ref, "CS", 0.3),
        L=parse_ref_value(ref, "L", 0.0),
        KE=FIXED_PARAMS["KE"],        # 直接从 FIXED 拿, 不参与率定
        xe=parse_ref_value(ref, "xe", 0.2),
    )
    # 保证三层张力水一致性
    p.WDM = p.WM - p.WUM - p.WLM
    if p.WDM < 5.0:
        p.WDM = 5.0
        p.WM = p.WUM + p.WLM + p.WDM
    return p


def get_calib_keys(mode: str) -> List[str]:
    """返回指定模式 (daily/event) 的率定键名列表。"""
    if mode == "daily":
        return list(DAILY_CALIB_KEYS)
    if mode == "event":
        return list(EVENT_CALIB_KEYS)
    raise ValueError(f"mode 必须是 'daily' 或 'event', 收到 {mode!r}")


def get_calib_bounds(mode: str) -> List[Tuple[float, float]]:
    """返回指定模式的率定边界 [(lo, hi), ...]。"""
    if mode == "daily":
        keys = DAILY_CALIB_KEYS
        return [DAILY_CALIB_BOUND[k] for k in keys]
    if mode == "event":
        keys = EVENT_CALIB_KEYS
        return [EVENT_CALIB_BOUND[k] for k in keys]
    raise ValueError(f"mode 必须是 'daily' 或 'event', 收到 {mode!r}")


def get_calib_ref(mode: str) -> Dict[str, Tuple[float, float]]:
    """返回指定模式的率定参考范围。"""
    if mode == "daily":
        return dict(DAILY_CALIB_REF)
    if mode == "event":
        return dict(EVENT_CALIB_REF)
    raise ValueError(f"mode 必须是 'daily' 或 'event', 收到 {mode!r}")


def make_bounds(keys, overrides: Dict[str, Tuple[float, float]] = None):
    """从 PARAM_BOUNDS 构造 ParameterBounds 列表, 允许 overrides 覆盖。"""
    overrides = overrides or {}
    result = []
    for k in keys:
        if k in overrides:
            lo, hi = overrides[k]
        else:
            lo, hi = DAILY_CALIB_BOUND.get(k) or EVENT_CALIB_BOUND.get(k) or (0.0, 1.0)
        result.append(ParameterBounds(name=k, min_value=float(lo),
                                      max_value=float(hi)))
    return result


# ─────────────────── DE 率定配置 (核心优化器配置) ───────────────────
# 设计依据:
#   - popsize: 标准 DE 推荐 ≥ 5 × n_params
#   - 收敛面平坦 → 可以降 iter 降 seed 大幅省时间
#   - 有 L-BFGS-B 精修兜底, DE 只做粗搜
DE_CONFIG_DAILY = {
    "popsize": 20,        # 6×3.3, Sobol 覆盖下足够
    "maxiter": 8,         # 8 代 (平坦, 不需要多代)
    "n_seeds": 3,         # 3 种子 (从 5 降, 省时间)
    "tol": 1e-6,
    "polish": False,      # 统一最后 L-BFGS-B 精修
    "init": "sobol",
    "early_stop_patience": 5,   # 连续 5 代无改进则停 (相对停滞 < 1e-4)
    "early_stop_min_delta": 1e-4,
}

DE_CONFIG_EVENT = {
    "popsize": 25,        # 保留, 多场联合需要大种群
    "maxiter": 15,        # 从 40 砍到 15 (省 ~60%)
    "n_seeds": 2,         # 从 5 砍到 2
    "tol": 1e-6,
    "polish": False,
    "init": "sobol",
    "early_stop_patience": 5,   # 连续 5 代无改进则停
    "early_stop_min_delta": 1e-4,
}


# ─────────────────── QUICK_MODE (汇报快跑用) ───────────────────
# True  → 用下面超小的 DE 配置, 总耗时 < 30s
# False → 走上面的标准配置 (n_seeds 减半, 早停控制总时间)
QUICK_MODE = True

# 标准配置, 但把 n_seeds 减半, 让早停更早触发
DE_CONFIG_DAILY_QUICK = {
    "popsize": 15,
    "maxiter": 5,
    "n_seeds": 2,             # 2 轮 (用户要求"几轮")
    "tol": 1e-5,
    "polish": False,
    "init": "sobol",
    "early_stop_patience": 3,
    "early_stop_min_delta": 1e-4,
}

DE_CONFIG_EVENT_QUICK = {
    "popsize": 18,
    "maxiter": 5,
    "n_seeds": 2,             # 2 轮
    "tol": 1e-5,
    "polish": False,
    "init": "sobol",
    "early_stop_patience": 3,
    "early_stop_min_delta": 1e-4,
}
